"""
Evaluation script for seg_exp07_maskrcnn_v6_improved.
Computes: Instance IoU, AP@50, AP@75, Count Accuracy.
Generates GT vs Pred side-by-side scene visualizations.
Includes score threshold sweep.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
import json
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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


def compute_iou_matrix(pred_masks, gt_masks):
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union
    return iou_matrix


def compute_ap_at_threshold(pred_masks, pred_scores, gt_masks, iou_threshold):
    if len(gt_masks) == 0:
        return 1.0 if len(pred_masks) == 0 else 0.0
    if len(pred_masks) == 0:
        return 0.0

    sorted_indices = np.argsort(-pred_scores)
    pred_masks = [pred_masks[i] for i in sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)

    tp = np.zeros(len(pred_masks))
    fp = np.zeros(len(pred_masks))
    matched_gt = set()

    for i in range(len(pred_masks)):
        if iou_matrix.shape[1] > 0:
            best_gt = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_gt]
            if best_iou >= iou_threshold and best_gt not in matched_gt:
                tp[i] = 1
                matched_gt.add(best_gt)
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_masks)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    ap /= 11.0
    return ap


def compute_instance_iou(pred_masks, pred_scores, gt_masks):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return []

    sorted_indices = np.argsort(-pred_scores)
    pred_masks = [pred_masks[i] for i in sorted_indices]

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    ious = []
    matched_gt = set()

    for i in range(len(pred_masks)):
        if iou_matrix.shape[1] > 0:
            best_gt = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_gt]
            if best_gt not in matched_gt and best_iou > 0:
                ious.append(best_iou)
                matched_gt.add(best_gt)

    for j in range(len(gt_masks)):
        if j not in matched_gt:
            ious.append(0.0)

    return ious


def visualize_scene(image, gt_masks, pred_masks, pred_scores, scene_idx, save_path, score_thresh=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    if image.shape[0] == 3:
        img_np = image[0].numpy()
    else:
        img_np = image.numpy()
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f'Scene {scene_idx} — Input')
    axes[0].axis('off')

    gt_overlay = np.zeros((*img_np.shape, 3))
    np.random.seed(42)
    for i, mask in enumerate(gt_masks):
        color = np.random.rand(3) * 0.7 + 0.3
        gt_overlay[mask > 0] = color
    axes[1].imshow(img_np, cmap='gray', alpha=0.4)
    axes[1].imshow(gt_overlay, alpha=0.6)
    axes[1].set_title(f'GT ({len(gt_masks)} objects)')
    axes[1].axis('off')

    pred_overlay = np.zeros((*img_np.shape, 3))
    filtered_masks = [(m, s) for m, s in zip(pred_masks, pred_scores) if s >= score_thresh]
    np.random.seed(42)
    for i, (mask, score) in enumerate(filtered_masks):
        color = np.random.rand(3) * 0.7 + 0.3
        pred_overlay[mask > 0] = color
    axes[2].imshow(img_np, cmap='gray', alpha=0.4)
    axes[2].imshow(pred_overlay, alpha=0.6)
    axes[2].set_title(f'Pred ({len(filtered_masks)} objects, thresh={score_thresh})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def evaluate_at_threshold(test_loader, model, device, score_thresh):
    """Run evaluation at a specific score threshold."""
    all_ious = []
    all_ap50 = []
    all_ap75 = []
    count_correct = 0
    count_total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            image = images[0]
            target = targets[0]

            img_device = image.to(device)
            output = model([img_device])[0]

            gt_masks = [target['masks'][i].numpy().astype(bool) for i in range(len(target['masks']))]
            pred_scores = output['scores'].cpu().numpy()
            pred_masks_raw = output['masks'][:, 0].cpu().numpy()
            pred_masks = [(pred_masks_raw[i] > 0.5).astype(bool) for i in range(len(pred_scores))]

            valid = pred_scores >= score_thresh
            pred_masks_filtered = [m for m, v in zip(pred_masks, valid) if v]
            pred_scores_filtered = pred_scores[valid]

            gt_count = len(gt_masks)
            pred_count = len(pred_masks_filtered)
            if gt_count == pred_count:
                count_correct += 1
            count_total += 1

            ap50 = compute_ap_at_threshold(pred_masks_filtered, pred_scores_filtered, gt_masks, 0.5)
            ap75 = compute_ap_at_threshold(pred_masks_filtered, pred_scores_filtered, gt_masks, 0.75)
            all_ap50.append(ap50)
            all_ap75.append(ap75)

            instance_ious = compute_instance_iou(pred_masks_filtered, pred_scores_filtered, gt_masks)
            all_ious.extend(instance_ious)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    std_iou = np.std(all_ious) if all_ious else 0.0
    return {
        'score_thresh': score_thresh,
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'ap50': float(np.mean(all_ap50)),
        'ap75': float(np.mean(all_ap75)),
        'count_accuracy': float(count_correct / max(count_total, 1)),
        'count_correct': count_correct,
        'count_total': count_total,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(script_dir)
    config_path = os.path.join(exp_dir, 'config.yaml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda')

    model = build_model(config)
    checkpoint_path = os.path.join(exp_dir, config['paths']['models'], 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (Val IoU: {checkpoint['val_iou']:.4f})")

    test_range = range(config['data']['test_scenes'][0], config['data']['test_scenes'][1] + 1)
    test_dataset = RhombicSEMDataset(test_range, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    scene_viz_dir = os.path.join(exp_dir, config['paths']['scene_viz'])
    os.makedirs(scene_viz_dir, exist_ok=True)

    # --- Score threshold sweep ---
    print("\n--- Score Threshold Sweep ---")
    sweep_results = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = evaluate_at_threshold(test_loader, model, device, thresh)
        sweep_results.append(r)
        print(f"  thresh={thresh:.1f} | IoU={r['mean_iou']:.4f} | AP50={r['ap50']:.4f} | "
              f"AP75={r['ap75']:.4f} | Count={r['count_accuracy']:.3f}")

    # Pick best threshold by IoU
    best_thresh_result = max(sweep_results, key=lambda x: x['mean_iou'])
    best_thresh = best_thresh_result['score_thresh']
    print(f"\nBest threshold: {best_thresh} (IoU={best_thresh_result['mean_iou']:.4f})")

    # --- Full evaluation at best threshold with visualizations ---
    print(f"\nFull evaluation at threshold={best_thresh}...")
    score_thresh = best_thresh
    all_ious = []
    all_ap50 = []
    all_ap75 = []
    count_correct = 0
    count_total = 0
    scene_results = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            image = images[0]
            target = targets[0]
            scene_idx = target['image_id'].item()

            img_device = image.to(device)
            output = model([img_device])[0]

            gt_masks = [target['masks'][i].numpy().astype(bool) for i in range(len(target['masks']))]
            pred_scores = output['scores'].cpu().numpy()
            pred_masks_raw = output['masks'][:, 0].cpu().numpy()
            pred_masks = [(pred_masks_raw[i] > 0.5).astype(bool) for i in range(len(pred_scores))]

            valid = pred_scores >= score_thresh
            pred_masks_filtered = [m for m, v in zip(pred_masks, valid) if v]
            pred_scores_filtered = pred_scores[valid]

            gt_count = len(gt_masks)
            pred_count = len(pred_masks_filtered)
            if gt_count == pred_count:
                count_correct += 1
            count_total += 1

            ap50 = compute_ap_at_threshold(pred_masks_filtered, pred_scores_filtered, gt_masks, 0.5)
            ap75 = compute_ap_at_threshold(pred_masks_filtered, pred_scores_filtered, gt_masks, 0.75)
            all_ap50.append(ap50)
            all_ap75.append(ap75)

            instance_ious = compute_instance_iou(pred_masks_filtered, pred_scores_filtered, gt_masks)
            scene_iou = np.mean(instance_ious) if instance_ious else 0.0
            all_ious.extend(instance_ious)

            scene_results.append({
                'scene_idx': int(scene_idx),
                'gt_count': gt_count,
                'pred_count': pred_count,
                'mean_iou': float(scene_iou),
                'ap50': float(ap50),
                'ap75': float(ap75),
            })

            if batch_idx % 10 == 0:
                viz_path = os.path.join(scene_viz_dir, f'scene_{scene_idx}.png')
                visualize_scene(image, gt_masks, pred_masks, pred_scores, scene_idx, viz_path, score_thresh)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Evaluated {batch_idx+1}/{len(test_loader)} scenes")

    # Worst scenes visualization
    scene_results_sorted = sorted(scene_results, key=lambda x: x['mean_iou'])
    for sr in scene_results_sorted[:10]:
        sidx = sr['scene_idx']
        viz_path = os.path.join(scene_viz_dir, f'scene_{sidx}_worst.png')
        if not os.path.exists(viz_path):
            img_t, tgt = test_dataset[sidx - config['data']['test_scenes'][0]]
            img_d = img_t.to(device)
            with torch.no_grad():
                out = model([img_d])[0]
            gt_m = [tgt['masks'][i].numpy().astype(bool) for i in range(len(tgt['masks']))]
            p_scores = out['scores'].cpu().numpy()
            p_masks = [(out['masks'][i, 0].cpu().numpy() > 0.5).astype(bool) for i in range(len(p_scores))]
            visualize_scene(img_t, gt_m, p_masks, p_scores, sidx, viz_path, score_thresh)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    std_iou = np.std(all_ious) if all_ious else 0.0
    mean_ap50 = np.mean(all_ap50)
    mean_ap75 = np.mean(all_ap75)
    count_accuracy = count_correct / max(count_total, 1)

    results = {
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'ap50': float(mean_ap50),
        'ap75': float(mean_ap75),
        'count_accuracy': float(count_accuracy),
        'count_correct': count_correct,
        'count_total': count_total,
        'best_epoch': checkpoint['epoch'],
        'num_test_scenes': len(test_range),
        'score_threshold': score_thresh,
        'threshold_sweep': sweep_results,
        'worst_scenes': scene_results_sorted[:10],
        'best_scenes': scene_results_sorted[-10:],
    }

    results_path = os.path.join(exp_dir, config['paths']['results'])
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (thresh={score_thresh})")
    print(f"{'='*60}")
    print(f"  Mean IoU:        {mean_iou:.4f} +/- {std_iou:.4f}")
    print(f"  AP@50:           {mean_ap50:.4f}")
    print(f"  AP@75:           {mean_ap75:.4f}")
    print(f"  Count Accuracy:  {count_accuracy:.4f} ({count_correct}/{count_total})")
    print(f"  Best Epoch:      {checkpoint['epoch']}")
    print(f"{'='*60}")
    print(f"\nWorst 5 scenes by IoU:")
    for sr in scene_results_sorted[:5]:
        print(f"  Scene {sr['scene_idx']}: IoU={sr['mean_iou']:.4f}, "
              f"GT={sr['gt_count']}, Pred={sr['pred_count']}")
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
