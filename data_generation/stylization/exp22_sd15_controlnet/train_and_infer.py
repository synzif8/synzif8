"""
Exp 15: ControlNet Fine-tune on SEM + UNet LoRA

Key idea: Fine-tune ControlNet so it learns "canny edge → SEM texture" mapping directly.
Then at inference, feed render's canny edges → ControlNet generates SEM-like output.
This avoids img2img's flat texture residual problem.

Phase 1: Fine-tune ControlNet on (SEM_canny, SEM) pairs + UNet LoRA
Phase 2: txt2img + ControlNet inference with render canny edges

Usage:
    CUDA_VISIBLE_DEVICES=2 python train_and_infer.py --phase both
"""

import argparse, os, random, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel, DDIMScheduler, DDPMScheduler,
    AutoencoderKL, UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class SEMControlNetDataset(Dataset):
    """Creates (canny_edge, SEM_image) pairs for ControlNet training."""
    def __init__(self, sem_dir, resolution=512, canny_low=20, canny_high=100):
        self.files = sorted([os.path.join(sem_dir, f) for f in os.listdir(sem_dir) if f.endswith('.png')])
        self.res = resolution
        self.cl = canny_low; self.ch = canny_high
        self.repeat = max(1, 200 // len(self.files))

    def __len__(self): return len(self.files) * self.repeat

    def __getitem__(self, idx):
        img = np.array(Image.open(self.files[idx % len(self.files)]).convert('L'), dtype=np.float32)
        h, w = img.shape; s = min(h, w)
        t = random.randint(0, h-s); l = random.randint(0, w-s)
        img = img[t:t+s, l:l+s]
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize((self.res, self.res), Image.LANCZOS))

        # Augment
        if random.random() > 0.5: img = np.flip(img, 1).copy()
        b = random.uniform(-0.15, 0.15) * 255
        c = 1 + random.uniform(-0.15, 0.15)
        img = np.clip((img.astype(np.float32) + b - img.mean()) * c + img.mean(), 0, 255).astype(np.uint8)

        # Canny edges (control signal)
        canny = cv2.Canny(img, self.cl, self.ch)

        # Normalize
        img_3ch = np.stack([img.astype(np.float32)]*3, axis=0)
        img_norm = (img_3ch / 127.5) - 1.0

        canny_3ch = np.stack([canny.astype(np.float32)]*3, axis=0)
        canny_norm = canny_3ch / 255.0  # [0, 1] for controlnet

        return torch.from_numpy(img_norm).float(), torch.from_numpy(canny_norm).float()


def train(device='cuda:0'):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(exp_dir, 'checkpoints'); os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading models...")
    scheduler = DDPMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae").to(device, torch.float16).eval()
    unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet").to(device, torch.float32)
    tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    text_enc = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder").to(device, torch.float16).eval()
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32).to(device)

    # Apply LoRA to UNet (reuse Exp 12 weights as starting point)
    lora_path = os.path.join(exp_dir, '..', 'exp21_sd15_lora', 'checkpoints', 'lora_final')
    if os.path.exists(lora_path):
        print(f"Loading pre-trained LoRA from {lora_path}")
        unet = PeftModel.from_pretrained(unet, lora_path)
        # Freeze UNet LoRA - only train ControlNet
        for p in unet.parameters(): p.requires_grad_(False)
    print(f"UNet trainable: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    # Make ControlNet trainable
    controlnet.train()
    cn_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"ControlNet trainable: {cn_params/1e6:.1f}M")

    # Text embeddings
    prompt = "high quality SEM micrograph, scanning electron microscopy, grayscale, crystalline particles with edge charging, grain texture"
    ti = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad(): text_emb = text_enc(ti.input_ids.to(device))[0].to(torch.float32)

    # Dataset
    ds = SEMControlNetDataset('../../SEM_dev ', 512, 20, 100)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    # Optimizer (only ControlNet)
    opt = torch.optim.AdamW(controlnet.parameters(), lr=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-7)

    print(f"Training ControlNet | {len(ds)} pairs | 500 epochs")
    logs = []

    for ep in range(1, 501):
        controlnet.train()
        eloss = 0

        for sem_img, canny_img in dl:
            sem_img, canny_img = sem_img.to(device), canny_img.to(device)

            with torch.no_grad():
                lat = vae.encode(sem_img.to(torch.float16)).latent_dist.sample().to(torch.float32) * vae.config.scaling_factor

            noise = torch.randn_like(lat)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy = scheduler.add_noise(lat, noise, t)

            # ControlNet forward
            down_samples, mid_sample = controlnet(
                noisy, t, encoder_hidden_states=text_emb,
                controlnet_cond=canny_img, return_dict=False
            )

            # UNet forward with ControlNet outputs
            pred = unet(
                noisy, t, encoder_hidden_states=text_emb,
                down_block_additional_residuals=[s.to(torch.float32) for s in down_samples],
                mid_block_additional_residual=mid_sample.to(torch.float32),
            ).sample

            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            opt.step()
            eloss += loss.item()

        sched.step()
        avg = eloss / len(dl); lr = sched.get_last_lr()[0]
        logs.append({"epoch": ep, "loss": avg, "lr": lr})
        if ep % 25 == 0 or ep == 1:
            print(f"Ep {ep}/500 | Loss: {avg:.6f} | LR: {lr:.2e}")
        if ep % 250 == 0:
            torch.save(controlnet.state_dict(), os.path.join(ckpt_dir, f"controlnet_ep{ep:04d}.pt"))

    torch.save(controlnet.state_dict(), os.path.join(ckpt_dir, "controlnet_final.pt"))
    with open(os.path.join(exp_dir, "train_log.json"), "w") as f: json.dump(logs, f, indent=2)
    print("Training done!")


def inference(device='cuda:0'):
    torch.manual_seed(42)
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(exp_dir, 'results', 'samples'); os.makedirs(rd, exist_ok=True)

    print("Loading pipeline...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    # Load fine-tuned weights
    cn_path = os.path.join(exp_dir, 'checkpoints', 'controlnet_final.pt')
    if os.path.exists(cn_path):
        sd = torch.load(cn_path, map_location='cpu')
        controlnet.load_state_dict(sd)
        print("Loaded fine-tuned ControlNet")

    # txt2img + ControlNet (NOT img2img!)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load UNet LoRA
    lora_path = os.path.join(exp_dir, '..', 'exp21_sd15_lora', 'checkpoints', 'lora_final')
    if os.path.exists(lora_path):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.to(device)

    prompt = "high quality SEM micrograph, scanning electron microscopy, grayscale, crystalline particles with edge charging effect, detailed grain texture, substrate background, scientific imaging"
    neg = "blurry, smooth, flat, pure black background, cartoon, colorful, low quality, CGI, synthetic, rendered"

    rdir = '../../dataset_v5'
    rf = sorted([f for f in os.listdir(rdir) if f.endswith('.png') and '_labeled' not in f and '_mask' not in f and '_edge' not in f and 'render_' in f])
    indices = np.linspace(0, len(rf)-1, 10, dtype=int)
    selected = [rf[i] for i in indices]
    sdir = '../../SEM_dev '
    sf = sorted([f for f in os.listdir(sdir) if f.endswith('.png')])

    for cn_scale in [0.5, 0.8, 1.0, 1.3]:
        tag = f"cn{cn_scale:.1f}"
        out = os.path.join(rd, tag); os.makedirs(out, exist_ok=True)
        print(f"\nControlNet scale={cn_scale}:")

        for i, fn in enumerate(selected):
            rg = np.array(Image.open(os.path.join(rdir, fn)).convert('L'))
            rg_r = np.array(Image.fromarray(rg).resize((512, 512), Image.LANCZOS))
            canny = cv2.Canny(rg_r, 20, 100)
            cp = Image.fromarray(np.stack([canny]*3, axis=-1))

            with torch.no_grad():
                res = pipe(prompt=prompt, negative_prompt=neg, image=cp,
                          num_inference_steps=50, guidance_scale=12.0,
                          controlnet_conditioning_scale=cn_scale,
                          generator=torch.Generator(device).manual_seed(42+i),
                          height=512, width=512).images[0]

            rg_out = np.array(res.convert('L'))
            ref = np.array(Image.open(os.path.join(sdir, sf[i%len(sf)])).convert('L'))
            ref_r = np.array(Image.fromarray(ref).resize((512,512), Image.LANCZOS))
            comp = np.concatenate([rg_r, rg_out, ref_r], axis=1)
            Image.fromarray(comp).save(os.path.join(out, f"compare_{fn}"))
            if i == 0: print(f"  {fn} done")

        print(f"  All -> {out}")

    print("\nAll done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="both", choices=["train","inference","both"])
    args = parser.parse_args()
    if args.phase in ("train","both"): train()
    if args.phase in ("inference","both"): inference()
