import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import geffnet
from dataset import SIIMISICDataset

img_size = 640

diagnosis2idx = {
    'AK': 0,
    'BCC': 1,
    'BKL': 2,
    'DF': 3,
    'SCC': 4,
    'VASC': 5,
    'melanoma': 6,
    'nevus': 7,
    'unknown': 8
}

transforms_test = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize()
])

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(backbone, pretrained=load_pretrained)
        self.dropout = nn.Dropout(0.5)

        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = self.myfc(self.dropout(x))
        return x

# n_test = 8
# df_test = df_test if not DEBUG else df_test.head(batch_size * 2)
# dataset_test = SIIMISICDataset(df_test, 'test', 'test', transform=transforms_val)
# test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)

# models = []
# for i_fold in range(5):
#     model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
#     model = model.to(device)
#     model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{i_fold}.pth')
#     state_dict = torch.load(model_file)
#     state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#     models.append(model)

OUTPUTS = []
PROBS = []

# with torch.no_grad():
#     for (data) in tqdm(test_loader):

#         if use_meta:
#             data, meta = data
#             data, meta = data.to(device), meta.to(device)
#             probs = torch.zeros((data.shape[0], out_dim)).to(device)
#             for I in range(n_test):
#                 l = model(get_trans(data, I), meta)
#                 probs += l.softmax(1)
#         else:
#             data = data.to(device)
#             probs = torch.zeros((data.shape[0], out_dim)).to(device)
#             for model in models:
#                 for I in range(n_test):
#                     l = model(get_trans(data, I))
#                     probs += l.softmax(1)

#         probs /= n_test * len(models)
#         PROBS.append(probs.detach().cpu())

# PROBS = torch.cat(PROBS).numpy()
# OUTPUTS = PROBS[:, 6]