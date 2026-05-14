"""DDP overfit runner — the single entry point for all baseline cycles.

Usage
-----
    torchrun --standalone --nproc_per_node=4 lineup/common/overfit_runner_ddp.py \\
        --model <candidate_name> \\
        --out-dir lineup/runs/<category>/<model>/overfit \\
        [--max-steps N] [--eval-every K] [--log-every K]

Each Candidate (see common/candidates/__init__.py) implements:
    name, category ∈ {"A", "B", "C"}
    crop_size, train_batch_size, max_steps
    build_model(device), build_optimizer(model, max_steps)
    forward_train(model_ddp, batch) -> {"loss": Tensor, ...}
    forward_eval(model_unwrapped, batch) -> {"R_pred","t_pred","scale_pred"} OR {"kp3d_pred"}

Dual-gate PASS: numeric thresholds + visual V1-V4 (the latter inspected later
via GT-leak-free grids the runner saves to ``<out_dir>/viz/``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

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
from common import geometry as geo
from common import viz_no_leak as vnl


PASS_THRESH = {
    "A": {"add_s": 4.5, "r_err": 2.0, "t_err": 1.5},
    "B": {"mpjpe": 2.0, "add_s": 4.5, "r_err": 2.0, "t_err": 1.5},
    "C": {"add_s": 9.0, "r_err": 5.0, "t_err": 3.0},
}


def setup_ddp() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    return rank, world, torch.device(f"cuda:{local}")


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def tensor_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    a = img_tensor.squeeze(0).detach().cpu().numpy()
    a = ((a + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)


def _decode_preds(out: dict, batch: dict) -> dict:
    """Normalize candidate output to (R_pred, t_pred, s_pred) for metrics."""
    if "R_pred" in out:
        R = out["R_pred"].detach().cpu().numpy()
        t = out["t_pred"].detach().cpu().numpy()
        s = (
            out["scale_pred"].detach().cpu().numpy()
            if "scale_pred" in out
            else batch["scale"].detach().cpu().numpy()
        )
        kp3d = None
    elif "kp3d_pred" in out:
        kp3d = out["kp3d_pred"].detach().cpu().numpy()
        B = kp3d.shape[0]
        Rs, ts, ss = [], [], []
        canon = batch["vertices_obj"].detach().cpu().numpy()
        for i in range(B):
            R_i, t_i, s_i = geo.procrustes_from_kp3d(kp3d[i], canon[i])
            Rs.append(R_i); ts.append(t_i); ss.append(s_i)
        R = np.stack(Rs); t = np.stack(ts); s = np.asarray(ss, dtype=np.float32)
    else:
        raise KeyError(f"candidate.forward_eval must return R_pred/t_pred or kp3d_pred, got {list(out)}")
    return {"R": R, "t": t, "s": s, "kp3d": kp3d}


def compute_metrics(preds: dict, batch: dict) -> list[dict]:
    R_b = batch["R"].detach().cpu().numpy()
    t_b = batch["t"].detach().cpu().numpy()
    s_b = batch["scale"].detach().cpu().numpy()
    canon = batch["vertices_obj"].detach().cpu().numpy()
    vcam_gt = batch["vertices_cam"].detach().cpu().numpy()
    B = R_b.shape[0]
    out = []
    for i in range(B):
        R_p, t_p, s_p = preds["R"][i], preds["t"][i], float(preds["s"][i])
        pred_verts = s_p * canon[i] @ R_p.T + t_p
        add = geo.sym_aware_vertex_error(pred_verts, vcam_gt[i], canon[i], R_b[i], t_b[i], float(s_b[i]))
        r_err = geo.sym_aware_rot_error_deg(R_p, R_b[i])
        t_err = float(np.linalg.norm(t_p - t_b[i]))
        rec = {
            "scene_id": int(batch["scene_id"][i].item()),
            "obj_id": int(batch["obj_id"][i].item()),
            "add_s": add["add_s_min"],
            "r_err": r_err,
            "t_err": t_err,
        }
        if preds["kp3d"] is not None:
            # MPJPE: per-vertex L2 to sym-aligned GT vertices
            errs = []
            for G in geo.O24_rotations():
                sym_gt = float(s_b[i]) * (canon[i] @ G.T) @ R_b[i].T + t_b[i]
                errs.append(np.linalg.norm(preds["kp3d"][i] - sym_gt, axis=1).mean())
            rec["mpjpe"] = float(min(errs))
        out.append(rec)
    return out


def aggregate_metrics(recs: list[dict]) -> dict[str, float]:
    if not recs:
        return {}
    keys = [k for k in recs[0] if k not in ("scene_id", "obj_id")]
    return {k: float(np.mean([r[k] for r in recs])) for k in keys}


def render_pred_grid(
    sample: dict, preds_i: dict, canonical_verts: np.ndarray, crop_affine: np.ndarray
) -> np.ndarray:
    img = tensor_to_bgr(sample["img_crop"])
    R_gt = sample["R"].detach().cpu().numpy()
    t_gt = sample["t"].detach().cpu().numpy()
    s_gt = float(sample["scale"])
    gt_panel = vnl.render_mesh_from_pose(img, R_gt, t_gt, s_gt, canonical_verts, crop_affine, color=(0, 255, 0))
    pred_panel = vnl.render_mesh_from_pose(
        img, preds_i["R"], preds_i["t"], float(preds_i["s"]), canonical_verts, crop_affine, color=(0, 165, 255)
    )
    # Error as mesh silhouette pred-vs-amodal-mask
    H, W = img.shape[:2]
    pred_verts = float(preds_i["s"]) * canonical_verts @ preds_i["R"].T + preds_i["t"]
    pred_2d = pred_verts[:, :2] @ crop_affine[:, :2].T + crop_affine[:, 2]
    pred_mask = np.zeros((H, W), np.uint8)
    hull = cv2.convexHull(pred_2d.astype(np.int32))
    cv2.fillConvexPoly(pred_mask, hull, 255)
    amodal = sample["mask_amodal"].squeeze(0).cpu().numpy().astype(bool)
    err = vnl.render_mask_error(pred_mask > 0, amodal)
    return vnl.build_2x2(img, gt_panel, pred_panel, err,
                         labels=("input", "gt", "pred", "mask err"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="candidate name in common.candidates.REGISTRY")
    ap.add_argument("--split", default="<PROJECT_ROOT>/baselines/phase0_common/splits/overfit.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--crop-size", type=int, default=None)
    ap.add_argument("--use-raw-render", action="store_true")
    args = ap.parse_args()

    rank, world, device = setup_ddp()

    candidate = load_candidate(args.model)
    crop_size = args.crop_size or candidate.crop_size
    bs = args.batch_size or candidate.train_batch_size
    max_steps = args.max_steps or candidate.max_steps

    out_dir = args.out_dir
    viz_dir = os.path.join(out_dir, "viz")
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        lock_path = os.path.join(out_dir, "RUNNING.lock")
        with open(lock_path, "w") as f:
            f.write(f"pid={os.getpid()} rank=0 world={world} start={time.time():.0f}\n")

    ds = InstanceDataset(
        split_path=args.split,
        crop_size=crop_size,
        use_styled=not args.use_raw_render,
    )
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=False) if world > 1 else None
    loader = DataLoader(
        ds,
        batch_size=bs,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    eval_sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None
    eval_loader = DataLoader(
        ds, batch_size=bs, sampler=eval_sampler, shuffle=False, collate_fn=collate, num_workers=0
    )

    model = candidate.build_model(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index])
    optim, sched = candidate.build_optimizer(model, max_steps)

    # Training
    step = 0
    train_log: list[dict] = []
    eval_log: list[dict] = []
    epoch = 0
    t0 = time.time()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= max_steps:
                break
            batch_d = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = candidate.forward_train(model, batch_d)
            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()

            # reduce loss for log
            loss_val = loss.detach()
            if world > 1:
                dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
            if is_main() and (step % args.log_every == 0 or step == max_steps - 1):
                rec = {"step": step, "loss": float(loss_val.item()), "lr": float(sched.get_last_lr()[0])}
                for k, v in out.items():
                    if k != "loss" and torch.is_tensor(v):
                        rec[k] = float(v.item())
                train_log.append(rec)
                print(f"[step {step:5d}] loss={rec['loss']:.5f} lr={rec['lr']:.2e}", flush=True)

            # eval
            if step > 0 and step % args.eval_every == 0:
                model.eval()
                all_recs: list[dict] = []
                with torch.no_grad():
                    for eb in eval_loader:
                        eb_d = {k: v.to(device, non_blocking=True) for k, v in eb.items()}
                        eo = candidate.forward_eval(model.module if world > 1 else model, eb_d)
                        preds = _decode_preds(eo, eb_d)
                        all_recs.extend(compute_metrics(preds, eb_d))
                # gather across ranks
                if world > 1:
                    buckets = [None for _ in range(world)]
                    dist.all_gather_object(buckets, all_recs)
                    all_recs = [r for bucket in buckets for r in bucket]
                if is_main():
                    agg = aggregate_metrics(all_recs)
                    agg["step"] = step
                    eval_log.append(agg)
                    s = " ".join(f"{k}={v:.3f}" for k, v in agg.items() if k != "step")
                    print(f"[eval {step:5d}] {s}", flush=True)
                model.train()
            step += 1
        epoch += 1

    # Final eval on full set at rank 0 for metrics + viz
    if is_main():
        model.eval()
        all_recs = []
        viz_samples = []
        canonical_np = None
        with torch.no_grad():
            # We run on rank 0 only to generate viz deterministically.
            ds_main = InstanceDataset(split_path=args.split, crop_size=crop_size, use_styled=not args.use_raw_render)
            for i in range(len(ds_main)):
                sample = ds_main[i]
                sample_d = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
                eo = candidate.forward_eval(model.module if world > 1 else model, sample_d)
                preds = _decode_preds(eo, sample_d)
                rec = compute_metrics(preds, sample_d)[0]
                all_recs.append(rec)
                preds_i = {"R": preds["R"][0], "t": preds["t"][0], "s": float(preds["s"][0])}
                if canonical_np is None:
                    canonical_np = sample["vertices_obj"].cpu().numpy()
                crop_affine_np = sample["crop_affine"].cpu().numpy()
                grid = render_pred_grid(sample, preds_i, canonical_np, crop_affine_np)
                cv2.imwrite(os.path.join(viz_dir, f"overfit_{int(sample['scene_id'].item()):04d}_{int(sample['obj_id'].item()):04d}.png"), grid)
                viz_samples.append(rec)

        agg = aggregate_metrics(all_recs)
        thresh = PASS_THRESH[candidate.category]
        numeric_pass = all(agg.get(k, 1e9) <= thresh[k] for k in thresh)
        metrics = {
            "candidate": candidate.name,
            "category": candidate.category,
            "n_instances": len(all_recs),
            "aggregate": agg,
            "per_instance": all_recs,
            "threshold": thresh,
            "numeric_pass": bool(numeric_pass),
            "train_log": train_log,
            "eval_log": eval_log,
            "world_size": world,
            "max_steps": max_steps,
            "crop_size": crop_size,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # rank-aware loss curve
        if eval_log:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            ax[0].plot([r["step"] for r in train_log], [r["loss"] for r in train_log], label="train")
            ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(True)
            for k in ("add_s", "r_err", "t_err", "mpjpe"):
                xs = [r["step"] for r in eval_log if k in r]
                ys = [r[k] for r in eval_log if k in r]
                if xs:
                    ax[1].plot(xs, ys, label=k)
            ax[1].set_xlabel("step"); ax[1].legend(); ax[1].grid(True); ax[1].set_yscale("log")
            fig.suptitle(f"{candidate.name} | numeric_pass={numeric_pass}")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "curves.png"), dpi=100)
            plt.close(fig)

        # Save weights
        state = model.module.state_dict() if world > 1 else model.state_dict()
        torch.save(state, os.path.join(out_dir, "weights_last.pt"))

        # STATUS.md shell (visual gate to be filled by reviewer)
        with open(os.path.join(out_dir, "STATUS.md"), "w") as f:
            f.write(f"# {candidate.name} — overfit status\n\n")
            f.write(f"- category: {candidate.category}\n")
            f.write(f"- world_size: {world}\n")
            f.write(f"- max_steps: {max_steps}\n")
            f.write(f"- numeric_gate: {'PASS' if numeric_pass else 'FAIL'}\n")
            f.write(f"- VISUAL_PASS: PENDING (reviewer must inspect viz/)\n\n")
            f.write("## Aggregate metrics\n")
            for k, v in agg.items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write(f"\nthreshold: {thresh}\n")

        os.remove(os.path.join(out_dir, "RUNNING.lock"))
        dt = time.time() - t0
        print(f"[done] runtime={dt:.1f}s numeric_pass={numeric_pass} -> {out_dir}/metrics.json")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
