"""Full epoch-based DDP runner — extends overfit_runner_ddp.py for Phase-1 training.

Reuses _decode_preds, compute_metrics, aggregate_metrics, render_pred_grid
from overfit_runner_ddp. Adds:
  - epoch loop with DistributedSampler.set_epoch
  - validation pass each epoch + early-stop on monitor metric
  - bf16 mixed precision (AMP) on Blackwell
  - torch.compile (with auto-fallback)
  - large per-rank batch + num_workers tuning
  - per-epoch fixed 5-instance viz (rank 0 only) using val_viz split
  - epoch_log.jsonl + STATUS.md

Usage:
  torchrun --standalone --nproc_per_node=4 lineup/common/full_runner_ddp.py \
      --model A1_gdrnet \
      --out-dir lineup/runs/full/A_appearance/A1_gdrnet \
      --train-split lineup/phase0_common/splits/train.txt \
      --val-split   lineup/phase0_common/splits/val.txt \
      --val-viz-split lineup/phase0_common/splits/val_viz.txt \
      [--max-epochs 200 --patience 20 --per-rank-batch 32 --num-workers 8 --no-compile]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.candidates import load_candidate
from common.dataset_instance import InstanceDataset, collate
from common.overfit_runner_ddp import (
    _decode_preds, compute_metrics, aggregate_metrics, render_pred_grid,
    setup_ddp, is_main, tensor_to_bgr, PASS_THRESH,
)
from common.geometry import O24_rotations  # noqa: F401 - keeps import-time check


# Default monitor key per category (improvement = lower is better)
MONITOR_DEFAULT = {"A": "add_s", "B": "mpjpe", "C": "add_s"}


def _build_loader(split: str, crop_size: int, batch: int, workers: int, world: int, rank: int, shuffle: bool):
    ds = InstanceDataset(split_path=split, crop_size=crop_size, use_styled=True)
    sampler = (
        DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False)
        if world > 1 else None
    )
    loader = DataLoader(
        ds,
        batch_size=batch,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    return ds, sampler, loader


def _amp_ctx(amp: str):
    if amp == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _eval_metrics(model, candidate, loader, device, world):
    """Run forward_eval over a loader and return per-instance records (rank 0 holds union)."""
    if hasattr(model, "module"):
        net = model.module
    else:
        net = model
    net.eval()
    recs: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            batch_d = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = candidate.forward_eval(net, batch_d)
            preds = _decode_preds(out, batch_d)
            recs.extend(compute_metrics(preds, batch_d))
    if world > 1:
        bucket = [None for _ in range(world)]
        dist.all_gather_object(bucket, recs)
        recs = [r for b in bucket for r in b]
    net.train()
    return recs


def _per_epoch_viz(candidate, model_unwrapped, viz_split: str, crop_size: int, device, out_dir: str, epoch: int):
    """Rank-0-only: forward_eval on the 5 fixed val_viz instances, write GT-leak-free grids."""
    ds = InstanceDataset(split_path=viz_split, crop_size=crop_size, use_styled=True)
    epoch_dir = os.path.join(out_dir, "viz", f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    canonical_np = None
    # Force eval mode: BN with running stats works at batch=1 + 1×1 spatial
    # (e.g., HccePose DeepLabV3 ASPP global-pool branch). Restore caller mode.
    was_training = model_unwrapped.training
    model_unwrapped.eval()
    try:
        with torch.no_grad():
            for i in range(len(ds)):
                sample = ds[i]
                sample_d = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
                out = candidate.forward_eval(model_unwrapped, sample_d)
                preds = _decode_preds(out, sample_d)
                if canonical_np is None:
                    canonical_np = sample["vertices_obj"].cpu().numpy()
                crop_affine_np = sample["crop_affine"].cpu().numpy()
                preds_i = {"R": preds["R"][0], "t": preds["t"][0], "s": float(preds["s"][0])}
                grid = render_pred_grid(sample, preds_i, canonical_np, crop_affine_np)
                sid = int(sample["scene_id"].item()); oid = int(sample["obj_id"].item())
                cv2.imwrite(os.path.join(epoch_dir, f"inst_{sid:04d}_{oid:04d}.png"), grid)
    finally:
        if was_training:
            model_unwrapped.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-split", default="<PROJECT_ROOT>/baselines/phase0_common/splits/train.txt")
    ap.add_argument("--val-split",   default="<PROJECT_ROOT>/baselines/phase0_common/splits/val.txt")
    ap.add_argument("--val-viz-split", default="<PROJECT_ROOT>/baselines/phase0_common/splits/val_viz.txt")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--monitor", default=None, help="metric key to monitor (lower=better). default per category")
    ap.add_argument("--per-rank-batch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--lr", type=float, default=None, help="override candidate's optimizer lr")
    ap.add_argument("--crop-size", type=int, default=None)
    ap.add_argument("--manifest", default="<PROJECT_ROOT>/baselines/phase0_common/splits/manifest.json")
    args = ap.parse_args()

    rank, world, device = setup_ddp()
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if is_main():
        print(f"[run] model={args.model} world={world} amp={args.amp} compile={not args.no_compile}", flush=True)

    candidate = load_candidate(args.model)
    crop_size = args.crop_size or candidate.crop_size
    cat = candidate.category

    # default per-rank batch by category
    default_bs = 32 if cat in ("A", "B") else 16
    bs = args.per_rank_batch or default_bs

    monitor_key = args.monitor or MONITOR_DEFAULT[cat]

    out_dir = args.out_dir
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "viz"), exist_ok=True)
        # Verify split manifest
        if os.path.exists(args.manifest):
            with open(args.manifest) as f:
                m = json.load(f)
            print(f"[run] using splits from {args.manifest} (seed={m['seed']})", flush=True)
        with open(os.path.join(out_dir, "RUNNING.lock"), "w") as f:
            f.write(f"pid={os.getpid()} time={time.time():.0f}\n")

    # Datasets / loaders
    _, train_sampler, train_loader = _build_loader(args.train_split, crop_size, bs, args.num_workers, world, rank, shuffle=True)
    _, val_sampler, val_loader     = _build_loader(args.val_split,   crop_size, bs, args.num_workers, world, rank, shuffle=False)

    # Model
    model = candidate.build_model(device)

    # torch.compile (best-effort)
    if not args.no_compile:
        try:
            model = torch.compile(model, mode="default")
            if is_main():
                print("[run] torch.compile OK", flush=True)
        except Exception as e:
            if is_main():
                print(f"[run] torch.compile fallback: {e}", flush=True)

    if world > 1:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    # Optimizer / scheduler — cosine T_max sized to expected-convergence epochs (not max_epochs).
    # Most candidates converge within 40-80 epochs; max_epochs is just a safety cap.
    steps_per_epoch = max(1, len(train_loader))
    cosine_target_epochs = min(args.max_epochs, 60)
    optim, sched = candidate.build_optimizer(model, cosine_target_epochs * steps_per_epoch)
    if args.lr is not None:
        for g in optim.param_groups:
            g["lr"] = args.lr

    # GradScaler for fp16; bf16 doesn't need it
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    log_path = os.path.join(out_dir, "epoch_log.jsonl") if is_main() else None
    best_metric = float("inf")
    best_epoch = -1
    patience_left = args.patience
    start_t = time.time()

    for epoch in range(args.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        ep_loss = 0.0; ep_n = 0

        for batch in train_loader:
            batch_d = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with _amp_ctx(args.amp):
                out = candidate.forward_train(model, batch_d)
                loss = out["loss"]
            if args.amp == "fp16":
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                optim.step()
            try:
                sched.step()
            except Exception:
                pass
            ep_loss += float(loss.detach().item()) * batch_d["img_crop"].shape[0]
            ep_n += batch_d["img_crop"].shape[0]

        # Reduce loss across ranks
        train_loss_t = torch.tensor([ep_loss, ep_n], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(train_loss_t, op=dist.ReduceOp.SUM)
        train_loss = float(train_loss_t[0].item() / max(train_loss_t[1].item(), 1.0))

        # Validation pass
        val_recs = _eval_metrics(model, candidate, val_loader, device, world)
        val_agg = aggregate_metrics(val_recs) if val_recs else {}
        val_metric = float(val_agg.get(monitor_key, float("inf")))

        improved = val_metric < best_metric
        if improved:
            best_metric = val_metric
            best_epoch = epoch
            patience_left = args.patience
            if is_main():
                state = (model.module if hasattr(model, "module") else model).state_dict()
                torch.save(state, os.path.join(out_dir, "weights_best.pt"))
        else:
            patience_left -= 1

        if is_main():
            state = (model.module if hasattr(model, "module") else model).state_dict()
            torch.save(state, os.path.join(out_dir, "weights_last.pt"))
            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_agg,
                "monitor": monitor_key,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "patience_left": patience_left,
                "lr": float(optim.param_groups[0]["lr"]),
                "elapsed_s": time.time() - start_t,
                "improved": bool(improved),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"[ep {epoch:03d}] train_loss={train_loss:.4f} {monitor_key}={val_metric:.4f} "
                f"best={best_metric:.4f}@{best_epoch} pat={patience_left} lr={rec['lr']:.2e} "
                f"t={rec['elapsed_s']:.0f}s",
                flush=True,
            )

            # Per-epoch fixed-instance viz
            try:
                _per_epoch_viz(
                    candidate,
                    model.module if hasattr(model, "module") else model,
                    args.val_viz_split,
                    crop_size,
                    device,
                    out_dir,
                    epoch,
                )
            except Exception as e:
                print(f"[ep {epoch:03d}] viz failed: {e}", flush=True)

        # Early stop
        if patience_left <= 0:
            if is_main():
                print(f"[run] early stop at epoch {epoch} (best @ {best_epoch})", flush=True)
            break

    # Wrap-up
    if is_main():
        runtime = time.time() - start_t
        with open(os.path.join(out_dir, "STATUS.md"), "w") as f:
            f.write(f"# {candidate.name} — full training status\n\n")
            f.write(f"- category: {cat}\n- world_size: {world}\n- max_epochs: {args.max_epochs}\n- patience: {args.patience}\n")
            f.write(f"- monitor: {monitor_key}\n- best_metric: {best_metric:.4f} @ epoch {best_epoch}\n")
            f.write(f"- runtime_s: {runtime:.0f}\n- amp: {args.amp}\n- per_rank_batch: {bs}\n")
        try:
            os.remove(os.path.join(out_dir, "RUNNING.lock"))
        except FileNotFoundError:
            pass

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
