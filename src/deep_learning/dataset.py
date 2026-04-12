"""
Dataset and data loading for image colorization.

Provides a PyTorch Dataset that loads RGB images, converts to Lab color space,
and returns L (input) and ab (target) channels for training/evaluation.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import color as skcolor


class ColorizationDataset(Dataset):
    """Dataset for image colorization training and evaluation.

    Loads RGB images, converts to CIE Lab, and returns normalized
    L channel (input) and ab channels (target).

    Args:
        image_dir: Path to directory containing images.
        input_size: Tuple (H, W) to resize images to.
        augment: Whether to apply data augmentation (flip, crop).
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, image_dir, input_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.input_size = input_size
        self.augment = augment

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # Load and convert to RGB
        img = Image.open(path).convert("RGB")
        img = img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        img_np = np.array(img)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np).copy()

        # Convert RGB [0, 255] -> Lab
        img_float = img_np.astype(np.float64) / 255.0
        lab = skcolor.rgb2lab(img_float)

        # Separate L and ab
        L = lab[:, :, 0]       # [0, 100]
        ab = lab[:, :, 1:]     # ~[-110, 110]

        # Normalize
        L_norm = L / 50.0 - 1.0        # [-1, 1]
        ab_norm = ab / 110.0           # ~[-1, 1]

        # Convert to tensors (C, H, W)
        L_tensor = torch.from_numpy(L_norm).float().unsqueeze(0)       # (1, H, W)
        ab_tensor = torch.from_numpy(ab_norm).float().permute(2, 0, 1) # (2, H, W)

        return {
            "L": L_tensor,
            "ab": ab_tensor,
            "path": path,
        }


def get_dataloaders(cfg):
    """Create train, validation, and test dataloaders from config.

    Args:
        cfg: dict with config (expects cfg["deep_learning"] and cfg["paths"]).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        Any loader may be None if the corresponding directory doesn't exist.
    """
    dl_cfg = cfg["deep_learning"]
    paths_cfg = cfg["paths"]
    input_size = tuple(cfg["image"]["input_size"])
    batch_size = dl_cfg["batch_size"]
    num_workers = dl_cfg.get("num_workers", 0)
    pin_memory = dl_cfg.get("pin_memory", True)

    coco_dir = paths_cfg["data_coco"]

    loaders = {}
    for split, augment in [("train2017", True), ("val2017", False), ("benchmark", False)]:
        split_dir = os.path.join(coco_dir, split)
        if os.path.isdir(split_dir) and len(os.listdir(split_dir)) > 0:
            dataset = ColorizationDataset(
                image_dir=split_dir,
                input_size=input_size,
                augment=augment,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train2017"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == "train2017"),
            )
        else:
            loaders[split] = None

    return loaders.get("train2017"), loaders.get("val2017"), loaders.get("benchmark")
