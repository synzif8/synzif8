"""
Exp 12: Strong LoRA (rank=64) + ControlNet Img2Img

Stronger SEM domain adaptation:
- LoRA rank 64 with attention + conv layers
- 1000 epochs training
- High strength (0.85) + high guidance (15.0) inference
- Raw render as input (no preprocessing)

Usage:
    CUDA_VISIBLE_DEVICES=2 python train_and_infer.py --config config.yaml --phase both
"""

import argparse, os, random, yaml, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel, DDIMScheduler, DDPMScheduler,
    AutoencoderKL, UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision.utils import save_image


class SEMDataset(Dataset):
    def __init__(self, sem_dir, resolution=512, augment=True, cfg=None):
        self.files = sorted([os.path.join(sem_dir, f) for f in os.listdir(sem_dir) if f.endswith('.png')])
        self.res = resolution; self.aug = augment; self.cfg = cfg
        self.repeat = max(1, 200 // len(self.files))

    def __len__(self): return len(self.files) * self.repeat

    def __getitem__(self, idx):
        img = np.array(Image.open(self.files[idx % len(self.files)]).convert('L'), dtype=np.float32)
        h, w = img.shape; s = min(h, w)
        t = random.randint(0, h-s); l = random.randint(0, w-s)
        img = img[t:t+s, l:l+s]
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize((self.res, self.res), Image.LANCZOS), dtype=np.float32)

        if self.aug:
            a = self.cfg.get('augmentation', {}) if self.cfg else {}
            if a.get('horizontal_flip') and random.random() > 0.5: img = np.flip(img, 1).copy()
            b = a.get('brightness_jitter', 0.15); c = a.get('contrast_jitter', 0.15)
            if b > 0: img = img + random.uniform(-b, b) * 255
            if c > 0: f = 1 + random.uniform(-c, c); img = (img - img.mean()) * f + img.mean()
            ns = a.get('gaussian_noise', 0)
            if ns > 0: img = img + np.random.randn(*img.shape).astype(np.float32) * random.uniform(0, ns)
            img = np.clip(img, 0, 255)

        img_3ch = np.stack([img]*3, axis=0)
        return torch.from_numpy((img_3ch / 127.5) - 1.0).float()


def train_lora(cfg):
    device = torch.device(cfg['device'])
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(exp_dir, 'checkpoints'); os.makedirs(ckpt_dir, exist_ok=True)
    lc = cfg['lora']

    print("Loading SD 1.5...")
    scheduler = DDPMScheduler.from_pretrained(cfg['models']['sd'], subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(cfg['models']['sd'], subfolder="vae").to(device, torch.float16).eval()
    unet = UNet2DConditionModel.from_pretrained(cfg['models']['sd'], subfolder="unet").to(device, torch.float32)
    tokenizer = CLIPTokenizer.from_pretrained(cfg['models']['sd'], subfolder="tokenizer")
    text_enc = CLIPTextModel.from_pretrained(cfg['models']['sd'], subfolder="text_encoder").to(device, torch.float16).eval()

    print("Applying LoRA (rank={})...".format(lc['rank']))
    lora_cfg = LoraConfig(r=lc['rank'], lora_alpha=lc['alpha'],
                          target_modules=lc['target_modules'], lora_dropout=lc['dropout'])
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    # Text embeddings
    ti = tokenizer(cfg['prompt'], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad(): text_emb = text_enc(ti.input_ids.to(device))[0].to(torch.float32)

    ds = SEMDataset(cfg['data']['train_sem'], lc['resolution'], True, cfg)
    dl = DataLoader(ds, batch_size=lc['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=lc['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=lc['epochs'], eta_min=1e-7)
    ga = lc.get('gradient_accumulation', 4)

    print(f"Training | {len(ds)} samples | {lc['epochs']} epochs")
    logs = []

    for ep in range(1, lc['epochs']+1):
        unet.train(); eloss = 0; opt.zero_grad()
        for step, batch in enumerate(dl):
            px = batch.to(device)
            with torch.no_grad():
                lat = vae.encode(px.to(torch.float16)).latent_dist.sample().to(torch.float32) * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=device).long()
            noisy = scheduler.add_noise(lat, noise, t)
            pred = unet(noisy, t, encoder_hidden_states=text_emb[:lat.shape[0]]).sample
            loss = F.mse_loss(pred, noise) / ga
            loss.backward(); eloss += loss.item() * ga
            if (step+1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step(); opt.zero_grad()
        if (step+1) % ga != 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step(); opt.zero_grad()

        sched.step()
        avg = eloss / len(dl); lr = sched.get_last_lr()[0]
        logs.append({"epoch": ep, "loss": avg, "lr": lr})
        if ep % 50 == 0 or ep == 1: print(f"Ep {ep}/{lc['epochs']} | Loss: {avg:.6f} | LR: {lr:.2e}")
        if ep % lc['save_every'] == 0:
            sp = os.path.join(ckpt_dir, f"lora_ep{ep:04d}")
            unet.save_pretrained(sp); print(f"  Saved {sp}")

    fp = os.path.join(ckpt_dir, "lora_final")
    unet.save_pretrained(fp)
    with open(os.path.join(exp_dir, "train_log.json"), "w") as f: json.dump(logs, f, indent=2)
    print(f"Done! Saved to {fp}")


def run_inference(cfg):
    device = torch.device(cfg['device']); torch.manual_seed(cfg['seed'])
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(exp_dir, 'results', 'samples'); os.makedirs(rd, exist_ok=True)
    ic = cfg['inference']

    print("Loading pipeline + LoRA...")
    cn = ControlNetModel.from_pretrained(cfg['models']['controlnet'], torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg['models']['sd'], controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    lp = os.path.join(exp_dir, 'checkpoints', 'lora_final')
    if os.path.exists(lp): pipe.unet = PeftModel.from_pretrained(pipe.unet, lp)
    else: print("WARNING: No LoRA!")
    pipe.to(device)

    rdir = cfg['data']['test_render']
    rf = sorted([f for f in os.listdir(rdir) if f.endswith('.png') and '_labeled' not in f and '_mask' not in f])
    idx = np.linspace(0, len(rf)-1, ic['num_test'], dtype=int)
    sel = [rf[i] for i in idx]
    sdir = cfg['data']['test_sem']
    sf = sorted([f for f in os.listdir(sdir) if f.endswith('.png')])

    print(f"Processing {len(sel)} renders (strength={ic['strength']}, guidance={ic['guidance_scale']})...")
    for i, fn in enumerate(sel):
        print(f"  [{i+1}/{len(sel)}] {fn}...", end=' ')
        rg = np.array(Image.open(os.path.join(rdir, fn)).convert('L'))
        rp = Image.fromarray(np.stack([rg]*3, axis=-1)).resize((512,512), Image.LANCZOS)
        r512 = np.array(rp.convert('L'))
        canny = cv2.Canny(r512, ic['canny_low'], ic['canny_high'])
        cp = Image.fromarray(np.stack([canny]*3, axis=-1))

        with torch.no_grad():
            res = pipe(prompt=cfg['prompt'], negative_prompt=cfg['negative_prompt'],
                      image=rp, control_image=cp, strength=ic['strength'],
                      num_inference_steps=ic['ddim_steps'], guidance_scale=ic['guidance_scale'],
                      controlnet_conditioning_scale=ic['controlnet_scale'],
                      generator=torch.Generator(device).manual_seed(cfg['seed']+i)).images[0]

        rg_out = np.array(res.convert('L'))
        Image.fromarray(rg_out).save(os.path.join(rd, f"sem_{fn}"))
        ref = np.array(Image.open(os.path.join(sdir, sf[i%len(sf)])).convert('L'))
        ref_r = np.array(Image.fromarray(ref).resize((512,512), Image.LANCZOS))
        comp = np.concatenate([r512, rg_out, ref_r], axis=1)
        Image.fromarray(comp).save(os.path.join(rd, f"compare_{fn}"))
        print("done")

    print(f"Results: {rd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", default="both", choices=["train","inference","both"])
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    if args.phase in ("train","both"): train_lora(cfg)
    if args.phase in ("inference","both"): run_inference(cfg)
