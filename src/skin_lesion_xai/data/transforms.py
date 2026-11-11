"""Albumentations pipelines. Geometric transforms are applied to the image and its mask
together (as in WA-NET's synchronised augmentation); colour transforms touch the image only.

Train augmentation follows GC-ViT-E §3.2 (random crops, flips, photometric jitter) and
Mahesh et al. §3.1 (rotation, flips, zoom). Eval = resize shorter side + center crop
(crop ratio 0.875, the timm/GC-ViT default `crop_pct`).
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_CROP_PCT = 0.875


def build_train_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-20, 20), scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.8),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.SmallestMaxSize(max_size=round(image_size / EVAL_CROP_PCT)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_transform(split: str, image_size: int = 224) -> A.Compose:
    return build_train_transform(image_size) if split == "train" else build_eval_transform(image_size)
