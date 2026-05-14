"""
dataset_v6 전체 변환 (멀티 GPU 병렬)
각 GPU가 독립 프로세스로 담당 범위를 처리

Usage:
    python inference_batch_v6.py
"""

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import numpy as np, os, cv2, sys
from PIL import Image


def run_worker(gpu_id, start_idx, end_idx, rdir, out_dir):
    # 이 워커가 사용할 GPU만 보이도록 제한 (다른 GPU에 더미 context 생성 방지)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
    from peft import PeftModel

    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES로 제한했으므로 항상 0
    torch.manual_seed(42)

    print(f'[GPU {gpu_id}] Loading pipeline...', flush=True)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=torch.float16)
    cn_path = '../exp22_sd15_controlnet/checkpoints/controlnet_final.pt'
    sd = torch.load(cn_path, map_location='cpu')
    controlnet.load_state_dict(sd)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5', controlnet=controlnet,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, '../exp21_sd15_lora/checkpoints/lora_final')
    pipe.to(device)
    print(f'[GPU {gpu_id}] Ready. Processing idx {start_idx}~{end_idx-1} ({end_idx-start_idx}장)', flush=True)

    prompt = 'high quality SEM micrograph, scanning electron microscopy, 8k resolution, hyper-realistic, grayscale, crystalline particles with strong bright edge charging effect, bright crystal edges, edge highlight, detailed grain texture, dark substrate background, dark background, scientific imaging, sharp details'
    neg = 'blurry, smooth, flat shading, light background, gray background, bright background, cartoon, colorful, low quality, CGI, synthetic, rendered, overexposed, no edge detail'

    cn_scale = 1.6
    guidance = 13.0
    fg_brightness = 1.2
    bg_brightness = 0.55
    steps = 60

    gen_w, gen_h = 1024, 704
    orig_w, orig_h = 1024, 686

    for i in range(start_idx, end_idx):
        fn = f'render_{i:04d}.png'
        edge_path = os.path.join(rdir, f'render_{i:04d}_edge.png')
        mask_path = os.path.join(rdir, f'render_{i:04d}_mask.png')

        if not os.path.exists(edge_path):
            continue

        edge = np.array(Image.open(edge_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))

        edge_r = np.array(Image.fromarray(edge).resize((gen_w, gen_h), Image.LANCZOS))
        cp = Image.fromarray(np.stack([edge_r] * 3, axis=-1))

        with torch.no_grad():
            res = pipe(prompt=prompt, negative_prompt=neg, image=cp,
                      num_inference_steps=steps, guidance_scale=guidance,
                      controlnet_conditioning_scale=cn_scale,
                      generator=torch.Generator(device).manual_seed(42 + i),
                      height=gen_h, width=gen_w).images[0]

        rg_out = np.array(res.convert('L').resize((orig_w, orig_h), Image.LANCZOS))

        # 밝기 보정
        img = rg_out.astype(np.float32)
        mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 3) / 255.0
        fg = img * fg_brightness
        bg = img * bg_brightness
        adjusted = fg * mask_soft + bg * (1 - mask_soft)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        Image.fromarray(adjusted).save(os.path.join(out_dir, f'sem_{i:04d}.png'))

        done = i - start_idx + 1
        total = end_idx - start_idx
        if done % 50 == 0 or done == 1:
            print(f'[GPU {gpu_id}] {done}/{total} ({done/total*100:.1f}%)', flush=True)

    print(f'[GPU {gpu_id}] 완료!', flush=True)


if __name__ == '__main__':
    rdir = '../../dataset_v6'
    out_dir = '../../dataset_v6_sem'
    os.makedirs(out_dir, exist_ok=True)

    total = 20000

    # GPU 0: 4000장, GPU 1,2,3: 나머지 균등 분배
    gpu0_count = 4000
    rest = total - gpu0_count
    per_rest = rest // 3

    assignments = [
        (0, 0, gpu0_count),
        (1, gpu0_count, gpu0_count + per_rest),
        (2, gpu0_count + per_rest, gpu0_count + per_rest * 2),
        (3, gpu0_count + per_rest * 2, total),
    ]

    print(f'총 {total}장 → dataset_v6_sem/')
    print(f'cn=1.6, g=13.0, fg=1.2, bg=0.55, steps=60, DDIM')
    for gpu_id, s, e in assignments:
        print(f'  GPU {gpu_id}: idx {s}~{e-1} ({e-s}장)')
    print()

    processes = []
    for gpu_id, s, e in assignments:
        p = mp.Process(target=run_worker, args=(gpu_id, s, e, rdir, out_dir))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f'\n전체 완료! {out_dir}에 저장됨.', flush=True)
