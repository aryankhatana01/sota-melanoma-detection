import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


class MelanomaDataset(Dataset):
    """
    Dataset class for melanoma dataset
    Args:
        df (pd.DataFrame): Dataframe containing image names and targets and meta features
        meta_features (list): List of meta features to be used
        transforms (albumentations.Compose): Albumentations transforms to be applied on the image
    
    Returns:
        dict: Returns a dictionary containing image, meta features and target
    """
    def __init__(self, df, meta_features=None, transforms=None):
        self.df = df.reset_index(drop=True)
        self.meta_features = meta_features
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.iloc[index].image_name
        img_path = os.path.join('../output/train', img_name + '.jpg')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            res = self.transforms(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.meta_features is not None:
            img = torch.tensor(image).float()
            meta_feats = torch.tensor(
                self.df.iloc[index][self.meta_features]
            ).float()
        else:
            img = torch.tensor(image).float()
            meta_feats = None
        return {
            'image': img,
            'meta_features': meta_feats,
            'target': torch.tensor(self.df.iloc[index].target).long()
        }

# To print a sample image
# plt.imshow(dataset[0]["image"].permute(1, 2, 0).numpy().astype(np.uint8))