import os
import cv2
import numpy as np
import pandas as pd
from functools import reduce
import albumentations as albu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.utils import ToCudaLoader
from src.dataset_dali import DaliLoader
from src.augmentations import get_aug

class DL_Proj(Dataset):
    def __init__(self, split="all", transform=None):
        """Args:
            split (str): one of `val`, `train`, `all`, `test`
            transform (albu.Compose): albumentation transformation for images
        """
        IMG_PATH = "workdir/images_512"
        MASK_PATH = "workdir/segm_masks"
        BORDER_PATH = "workdir/border_masks"
        ids = [int(i.split('.')[0]) for i in os.listdir(IMG_PATH)]
        if split == "train":
            ids = [i for i in ids if i % 10 > 1]
        elif split == "val":
            ids = [i for i in ids if i % 10 == 0]
        elif split == "test":
            ids = [i for i in ids if i % 10 == 1]
        
        self.img_ids = [f"{IMG_PATH}/{i}.png" for i in ids]
        self.mask_ids = [f"{MASK_PATH}/{i}.png" for i in ids]
        self.border_ids = [f"{BORDER_PATH}/{i}.png" for i in ids]
        self.transform = albu.Compose([]) if transform is None else transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_ids[idx], cv2.IMREAD_GRAYSCALE)
        border = cv2.imread(self.border_ids[idx], cv2.IMREAD_GRAYSCALE)
        mask = np.stack([border, mask], axis=2)
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented["image"], augmented["mask"] / 255.0
        return aug_img, aug_mask

def get_dataloaders(datasets, augmentation="medium", batch_size=16, size=384, val_size=384, buildings_only=False):
    """Returns:
    train_dataloader, val_dataloader
    """

    ## get augmentations
    train_aug = get_aug(augmentation, size=size)
    val_aug = get_aug("val", size=val_size)

    # get datasets
    val_datasets = []
    train_datasets = []
    if "tier1" in datasets:
        val_datasets.append(
            OpenCitiesDataset(
                split="val",
                transform=val_aug,
                imgs_path="data/tier_1-images-512",
                masks_path="data/tier_1-masks-512",
                buildings_only=buildings_only,
            )
        )
        train_datasets.append(
            OpenCitiesDataset(
                split="train",
                transform=train_aug,
                imgs_path="data/tier_1-images-512",
                masks_path="data/tier_1-masks-512",
                buildings_only=buildings_only,
            )
        )
    if "tier2" in datasets:
        val_datasets.append(
            OpenCitiesDataset(
                split="val",
                transform=val_aug,
                imgs_path="data/tier_2-images-512",
                masks_path="data/tier_2-masks-512",
                buildings_only=buildings_only,
            )
        )
        train_datasets.append(
            OpenCitiesDataset(
                split="train",
                transform=train_aug,
                imgs_path="data/tier_2-images-512",
                masks_path="data/tier_2-masks-512",
                buildings_only=buildings_only,
            )
        )
    if "inria" in datasets:
        val_datasets.append(
            InriaTilesDataset(
                split="val",
                transform=val_aug,
                buildings_only=buildings_only
            )
        )
        train_datasets.append(
            InriaTilesDataset(
                split="train",
                transform=train_aug,
                buildings_only=buildings_only,
            )
        )
    if "dl_proj" in datasets:
        val_datasets.append(DL_Proj(split="val", transform=val_aug))
        train_datasets.append(DL_Proj(split="train", transform=val_aug))

    if "inria_dali" in datasets:
        train_loader = DaliLoader(True, batch_size, size)
        val_loader = DaliLoader(False, batch_size, val_size)
        print(f"\nUsing datasets: {datasets}. Train size: {len(train_loader) * batch_size}. Val size {len(val_loader) * batch_size}.")
        return train_loader, val_loader

    # concat all datasets into one
    val_dtst = reduce(lambda x, y: x + y, val_datasets)
    val_dtld = DataLoader(val_dtst, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_dtld = ToCudaLoader(val_dtld)

    train_dtst = reduce(lambda x, y: x + y, train_datasets)
    # without `drop_last` last batch consists of 1 element and BN fails
    train_dtld = DataLoader(train_dtst, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    train_dtld = ToCudaLoader(train_dtld)

    print(f"\nUsing datasets: {datasets}. With {augmentation} augmentation. Train size: {len(train_dtst)}. Val size {len(val_dtst)}.")
    return train_dtld, val_dtld



