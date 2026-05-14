"""
Training script for Mask R-CNN on SEM-like rhombic object images.
seg_exp07: dataset_v6 improved — augmentation, cosine LR, instance-level val IoU.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 src/train.py
"""

import os
import sys
import time
import json
import math
import yaml
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_dataset import RhombicSEMDataset, collate_fn


def build_model(config):
    mc = config['model']
    model = maskrcnn_resnet50_fpn_v2(
        weights='DEFAULT',
        rpn_pre_nms_top_n_train=mc.get('rpn_pre_nms_top_n_train', 2000),
        rpn_post_nms_top_n_train=mc.get('rpn_post_nms_top_n_train', 2000),
        rpn_pre_nms_top_n_test=mc.get('rpn_pre_nms_top_n_test', 1000),
        rpn_post_nms_top_n_test=mc.get('rpn_post_nms_top_n_test', 1000),
        box_nms_thresh=mc.get('box_nms_thresh', 0.5),
        box_detections_per_img=mc.get('box_detections_per_img', 100),
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, mc['num_classes'])
    in_fm = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, mc['num_classes'])
    return model


def compute_instance_iou(pred_masks, pred_scores, gt_masks, score_thresh=0.5):
    """Instance-level IoU via greedy matching."""
    # Filter by score
    valid = pred_scores >= score_thresh
    pred_masks = [m for m, v in zip(pred_masks, valid) if v]
    pred_scores_f = pred_scores[valid]

    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return [], len(gt_masks), len(pred_masks)

    # Sort by score descending
    order = np.argsort(-pred_scores_f)
    pred_masks = [pred_masks[i] for i in order]

    # Greedy matching
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            inter = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
            if union > 0:
                iou_matrix[i, j] = inter / union

    ious = []
    matched_gt = set()
    for i in range(n_pred):
        if n_gt == 0:
            break
        best_gt = np.argmax(iou_matrix[i])
        best_iou = iou_matrix[i, best_gt]
        if best_gt not in matched_gt and best_iou > 0.1:
            ious.append(best_iou)
            matched_gt.add(best_gt)

    # Unmatched GT → IoU 0
    for j in range(n_gt):
        if j not in matched_gt:
            ious.append(0.0)

    return ious, n_gt, len(pred_masks)


def compute_val_metrics(model, val_loader, device, max_batches=32):
    """Compute validation loss and instance-level IoU in a single pass.
    Uses a subset (max_batches) to keep validation fast during training."""
    model.eval()
    total_loss = 0.0
    all_ious = []
    count_correct = 0
    count_total = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            if num_batches >= max_batches:
                break

            images_dev = [img.to(device) for img in images]
            targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get losses (temporarily switch to train mode)
            model.train()
            loss_dict = model(images_dev, targets_dev)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()

            # Get predictions
            outputs = model(images_dev)
            for out, tgt in zip(outputs, targets):
                gt_masks = [tgt['masks'][i].numpy().astype(bool) for i in range(len(tgt['masks']))]
                pred_scores = out['scores'].cpu().numpy()
                pred_masks = [(out['masks'][i, 0].cpu().numpy() > 0.5).astype(bool)
                              for i in range(len(pred_scores))]

                ious, n_gt, n_pred = compute_instance_iou(pred_masks, pred_scores, gt_masks)
                all_ious.extend(ious)

                if n_gt == n_pred:
                    count_correct += 1
                count_total += 1

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    count_acc = count_correct / max(count_total, 1)
    return avg_loss, mean_iou, count_acc


def warmup_cosine_lr(optimizer, epoch, iter_in_epoch, iters_per_epoch, config):
    """Warmup + cosine annealing LR schedule."""
    tc = config['train']
    warmup_iters = tc.get('warmup_iters', 1000)
    total_iters = tc['epochs'] * iters_per_epoch
    current_iter = (epoch - 1) * iters_per_epoch + iter_in_epoch

    if current_iter < warmup_iters:
        warmup_factor = tc.get('warmup_factor', 0.001)
        alpha = current_iter / warmup_iters
        lr = tc['lr'] * (warmup_factor * (1 - alpha) + alpha)
    else:
        progress = (current_iter - warmup_iters) / max(total_iters - warmup_iters, 1)
        lr = tc['lr'] * 0.5 * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    # Increase NCCL timeout for validation on rank 0
    from datetime import timedelta
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    is_main = (local_rank == 0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(script_dir)
    config_path = os.path.join(exp_dir, 'config.yaml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_dir = os.path.join(exp_dir, config['paths']['models'])
    tb_dir = os.path.join(exp_dir, config['paths']['tensorboard'])
    if is_main:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)

    # Dataset with augmentation
    train_range = range(config['data']['train_scenes'][0], config['data']['train_scenes'][1] + 1)
    val_range = range(config['data']['val_scenes'][0], config['data']['val_scenes'][1] + 1)

    aug_config = config.get('augmentation', {})
    train_dataset = RhombicSEMDataset(train_range, augment=True, aug_config=aug_config)
    val_dataset = RhombicSEMDataset(val_range, augment=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config['train']['batch_size'],
        sampler=train_sampler, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['train']['batch_size'],
        shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    model = build_model(config)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer (no scheduler — LR handled manually)
    tc = config['train']
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=tc['lr'],
        momentum=tc.get('momentum', 0.9),
        weight_decay=tc.get('weight_decay', 0.0005)
    )

    writer = SummaryWriter(log_dir=tb_dir) if is_main else None

    epochs = tc['epochs']
    best_iou = 0.0
    best_epoch = 0
    iters_per_epoch = len(train_loader)

    if is_main:
        print(f"Training {config['experiment']}")
        print(f"  Train: {len(train_range)} scenes, Val: {len(val_range)} scenes")
        print(f"  Epochs: {epochs}, Batch: {tc['batch_size']} x {dist.get_world_size()} GPUs")
        print(f"  LR: {tc['lr']} (warmup {tc.get('warmup_iters', 1000)} iters + cosine)")
        print(f"  Augmentation: {aug_config}")
        print(f"  Iters/epoch: {iters_per_epoch}")
        print()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        num_batches = 0
        current_lr = tc['lr']

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Update LR
            current_lr = warmup_cosine_lr(optimizer, epoch, batch_idx, iters_per_epoch, config)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            train_loss += losses.item()
            num_batches += 1

            if is_main and (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{iters_per_epoch} | "
                      f"Loss: {losses.item():.4f} | LR: {current_lr:.6f}")

        avg_train_loss = train_loss / max(num_batches, 1)

        if is_main:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Loss_detail/{k}', v.item(), epoch)

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            if is_main:
                val_model = model.module
                val_loss, val_iou, val_count_acc = compute_val_metrics(val_model, val_loader, device)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Metrics/val_instance_iou', val_iou, epoch)
                writer.add_scalar('Metrics/val_count_acc', val_count_acc, epoch)

                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | IoU: {val_iou:.4f} | "
                      f"Count: {val_count_acc:.3f} | LR: {current_lr:.6f} | {epoch_time:.0f}s")

                if val_iou > best_iou:
                    best_iou = val_iou
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_iou': val_iou,
                        'val_loss': val_loss,
                        'val_count_acc': val_count_acc,
                    }, os.path.join(model_dir, 'best_model.pth'))
                    print(f"  >>> New best (IoU: {val_iou:.4f}, Count: {val_count_acc:.3f})")

            dist.barrier()
        else:
            if is_main:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | "
                      f"LR: {current_lr:.6f} | {epoch_time:.0f}s")

        # Checkpoint every 30 epochs
        if is_main and epoch % 30 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(model_dir, f'checkpoint_epoch{epoch}.pth'))

    if is_main:
        writer.close()
        print(f"\nTraining complete. Best Instance IoU: {best_iou:.4f} at epoch {best_epoch}")

        summary = {
            'best_epoch': best_epoch,
            'best_val_iou': float(best_iou),
            'total_epochs': epochs,
        }
        with open(os.path.join(exp_dir, 'train_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
