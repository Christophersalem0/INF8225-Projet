"""
utils/dataloader.py  —  INF8225 Équipe 5
DataLoader unifié pour les 6 datasets du projet.

Tous les datasets sont ramenés à la même structure canonique :
    data/<NOM>/train/images/  +  train/masks/
    data/<NOM>/val/images/    +  val/masks/
    data/<NOM>/test/images/   +  test/masks/

SegDataset gère le chargement, les augmentations (Albumentations)
et la binarisation adaptive des masques.
"""

import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Normalisation ImageNet (RGB) et grayscale
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_GRAY_MEAN     = [0.5]
_GRAY_STD      = [0.229]

# Extensions d'images acceptées
_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')


class SegDataset(data.Dataset):
    """
    Dataset de segmentation binaire générique.

    Paramètres
    ----------
    image_dir   : dossier contenant les images
    mask_dir    : dossier contenant les masques binaires
    img_size    : taille cible après resize (carré)
    split       : 'train' | 'val' | 'test'
    augment     : True → augmentations d'entraînement
    rgb         : True → image RGB, False → grayscale
    """
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 352,
                 split: str = 'train', augment: bool = True, rgb: bool = True):
        self.img_size = img_size
        self.split    = split
        self.rgb      = rgb

        images = sorted(Path(image_dir).glob('*'))
        masks  = sorted(Path(mask_dir).glob('*'))
        self.image_paths = [p for p in images if p.suffix.lower() in _IMG_EXTS]
        self.mask_paths  = [p for p in masks  if p.suffix.lower() in _IMG_EXTS]
        self._filter_existing()

        mean, std = (_IMAGENET_MEAN, _IMAGENET_STD) if rgb else (_GRAY_MEAN, _GRAY_STD)

        base_transforms = [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
        ] + base_transforms

        self.transform = A.Compose(aug_transforms if (split == 'train' and augment)
                                   else base_transforms)

    def _filter_existing(self):
        valid = [(i, m) for i, m in zip(self.image_paths, self.mask_paths)
                 if i.exists() and m.exists()]
        self.image_paths, self.mask_paths = zip(*valid) if valid else ([], [])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lecture image
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if self.rgb else cv2.COLOR_BGR2GRAY)

        # Lecture masque (grayscale)
        mask_raw = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        # Transformation
        aug = self.transform(image=img, mask=mask_raw)
        image = aug['image']    # Tensor (C, H, W)
        mask  = aug['mask']     # Tensor (H, W) float

        # Binarisation adaptive : gère {0,255} ou {0,1,2,...}
        mask = (mask > 20).long() if mask.max() > 127 else (mask >= 1).long()
        mask = mask.unsqueeze(0)  # (1, H, W)

        if self.split == 'train':
            return image, mask

        # Pour val/test : retourne aussi la taille originale et le nom
        orig_w, orig_h = Image.open(self.mask_paths[idx]).size
        name = self.image_paths[idx].stem + '.png'
        return image, mask, (orig_h, orig_w), name


def build_loader(image_dir: str, mask_dir: str, batch_size: int,
                 img_size: int = 352, split: str = 'train',
                 augment: bool = True, rgb: bool = True,
                 num_workers: int = 4) -> data.DataLoader:
    """Construit un DataLoader à partir de deux dossiers image/mask."""
    dataset = SegDataset(image_dir, mask_dir, img_size, split, augment, rgb)
    return data.DataLoader(dataset, batch_size=batch_size,
                           shuffle=(split == 'train'),
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=(split == 'train'))
