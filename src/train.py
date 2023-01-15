import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import GradualWarmupSchedulerV2, get_transforms, get_meta_data
from config import Config
from dataset import MelanomaDataset
from models import Effnet_Melanoma

device = Config.device

def train_epoch(model, loader, optimizer, criterion, scheduler):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for data in bar:

        optimizer.zero_grad()
        
        if Config.use_meta:
            img = data['image']
            meta = data['meta_features']
            target = data['target']
            img, meta, target = img.to(device), meta.to(device), target.to(device)
            logits = model(img, meta)
        else:
            img, target = img.to(device), target.to(device)
            logits = model(data)        
        
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss

def get_trans(img, I):

    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, criterion, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for data in tqdm(loader):
            
            if Config.use_meta:
                img = data['image']
                meta = data['meta_features']
                target = data['target']
                img, meta, target = img.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((img.shape[0], 4)).to(device)
                probs = torch.zeros((img.shape[0], 4)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)

            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        return val_loss, acc


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val):

    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    dataset_train = MelanomaDataset(df_train, meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    model = Effnet_Melanoma(
        Config.model_name, 
        out_dim=Config.out_dim, 
        n_meta_features=n_meta_features, 
        n_meta_dim=[128, 32], 
        pretrained=True
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    model_file  = os.path.join('saved_models/', f'{Config.model_name}_best_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, Config.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)    

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, Config.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler_warmup)
        val_loss, acc = val_epoch(model, valid_loader, is_ext=df_valid['is_ext'].values)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}.'
        print(content)

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() # bug workaround   

    torch.save(model.state_dict(), model_file)


def main():

    df = pd.read_csv("../input/train_folds.csv")
    df_train, meta_features, n_meta_features = get_meta_data(df)

    transforms_train, transforms_val = get_transforms(Config.image_size)

    for fold in range(5):
        run(fold, df_train, meta_features, n_meta_features, transforms_train, transforms_val)

if __name__ == "__main__":
    main()