"""
Dataset for Rhombic Object Instance Segmentation using SEM-like images.
seg_exp07: dataset_v6 with data augmentation.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance


SEM_DIR = '<DATA_ROOT>/dataset_v6_sem'
MASK_DIR = '<DATA_ROOT>/dataset_v6'


class RhombicSEMDataset(Dataset):
    def __init__(self, scene_indices, augment=False, aug_config=None):
        self.scene_indices = list(scene_indices)
        self.augment = augment
        self.aug_config = aug_config or {}

    def __len__(self):
        return len(self.scene_indices)

    def _get_sem_path(self, scene_idx):
        return os.path.join(SEM_DIR, f'sem_{scene_idx:04d}.png')

    def _get_mask_dir(self, scene_idx):
        return os.path.join(MASK_DIR, f'render_{scene_idx:04d}_masks')

    def _apply_augmentation(self, img_pil, masks, boxes):
        """Apply augmentations to image, masks, and boxes together."""
        w, h = img_pil.size

        # Horizontal flip
        if random.random() < self.aug_config.get('horizontal_flip', 0):
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            masks = [np.fliplr(m).copy() for m in masks]
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]

        # Vertical flip
        if random.random() < self.aug_config.get('vertical_flip', 0):
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            masks = [np.flipud(m).copy() for m in masks]
            boxes = [[x1, h - y2, x2, h - y1] for x1, y1, x2, y2 in boxes]

        # Brightness jitter
        br = self.aug_config.get('brightness_range', None)
        if br is not None:
            factor = random.uniform(br[0], br[1])
            img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)

        # Contrast jitter
        cr = self.aug_config.get('contrast_range', None)
        if cr is not None:
            factor = random.uniform(cr[0], cr[1])
            img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)

        return img_pil, masks, boxes

    def __getitem__(self, idx):
        scene_idx = self.scene_indices[idx]

        # Load SEM-like image (Grayscale L, 1024x686)
        sem_path = self._get_sem_path(scene_idx)
        img_pil = Image.open(sem_path).convert('L')

        # Load per-object visible masks
        mask_dir = self._get_mask_dir(scene_idx)
        visible_files = sorted([
            f for f in os.listdir(mask_dir) if f.endswith('_visible.png')
        ])

        masks = []
        boxes = []
        for vf in visible_files:
            mask = np.array(Image.open(os.path.join(mask_dir, vf)))
            binary_mask = (mask > 0).astype(np.uint8)

            if binary_mask.sum() == 0:
                continue

            pos = np.where(binary_mask)
            ymin = pos[0].min()
            ymax = pos[0].max()
            xmin = pos[1].min()
            xmax = pos[1].max()

            if ymax <= ymin or xmax <= xmin:
                continue

            masks.append(binary_mask)
            boxes.append([xmin, ymin, xmax, ymax])

        # Apply augmentation
        if self.augment and len(masks) > 0:
            img_pil, masks, boxes = self._apply_augmentation(img_pil, masks, boxes)

        # Convert image to tensor
        arr = np.array(img_pil).astype(np.float32) / 255.0

        # Add gaussian noise
        if self.augment:
            noise_std = self.aug_config.get('gaussian_noise_std', 0)
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 1.0)

        img_tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).float()

        num_objs = len(masks)

        if num_objs == 0:
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'masks': torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.uint8),
                'image_id': torch.tensor([scene_idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64),
            }
        else:
            masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones(num_objs, dtype=torch.int64)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])

            target = {
                'boxes': boxes_tensor,
                'labels': labels,
                'masks': masks_tensor,
                'image_id': torch.tensor([scene_idx]),
                'area': area,
                'iscrowd': torch.zeros(num_objs, dtype=torch.int64),
            }

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
