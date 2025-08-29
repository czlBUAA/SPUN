'''
MIT License

Copyright (c) 2021 SLAB Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import torch

from .Park2019KRNDataset import Park2019KRNDataset
from .SPNDataset         import SPNDataset
from .transforms         import build_transforms
from torch.utils.data import Subset
def build_dataset(cfg, is_train=True, is_source=True, load_labels=True):

    transforms = build_transforms(cfg.model_name, cfg.input_shape, is_train=is_train)

    dataset = Park2019KRNDataset(cfg, transforms, is_train, is_source, load_labels)

    return dataset

def make_dataloader(cfg, is_train=True, is_source=True, is_test=False, load_labels=True):

    if is_train:
        images_per_gpu = cfg.batch_size
        shuffle = True
        num_workers = cfg.num_workers
    else:
        images_per_gpu = cfg.batch_size
        shuffle = False
        num_workers = 1
    if is_test:
        images_per_gpu = cfg.batch_size
        shuffle = True
        num_workers = 1

    dataset = build_dataset(cfg, is_train, is_source, load_labels)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
        )
    return data_loader
def make_dataloader1(cfg, is_train=True, is_source=True, is_test=False, load_labels=True):

    if is_train:
        images_per_gpu = cfg.batch_size
        shuffle = True
        num_workers = cfg.num_workers
    else:
        images_per_gpu = cfg.batch_size
        shuffle = False
        num_workers = 1
    if is_test:
        images_per_gpu = cfg.batch_size
        shuffle = False
        num_workers = 1
    dataset = build_dataset(cfg, is_train, is_source, load_labels)
    # dataset_size = len(dataset)
    # split = int(0.8 * dataset_size)

    # # # 前90%为训练集，后10%为验证集
    # train_indices = list(range(0, split))
    # val_indices = list(range(split, dataset_size))
    
    #train_dataset = Subset(dataset, train_indices)
    #val_dataset = Subset(dataset, val_indices)
    df = pd.read_csv("/media/computer/study/CZL/Bingham-Guass/speedplus/sunlamp/cv_indices.csv")

    # 取出第 i 折 (比如 i=0)
    i   = 0
    train_indices = eval(df.loc[i, "train_indices"])  # 转成 list
    val_indices = eval(df.loc[i, "val_indices"])

    # 构造数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=images_per_gpu,
    shuffle=shuffle,        # 训练集可设置为 True 以便打乱数据
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=images_per_gpu,
    shuffle=False,          # 验证集通常不需要打乱
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
    )
    return train_loader, val_loader
def make_dataloader2(cfg, is_train=True, is_source=True, is_test=False, load_labels=True):

    if is_train:
        images_per_gpu = cfg.batch_size
        shuffle = True
        num_workers = cfg.num_workers
    else:
        images_per_gpu = cfg.batch_size
        shuffle = False
        num_workers = 1
    if is_test:
        images_per_gpu = cfg.batch_size
        shuffle = False
        num_workers = 1
    dataset = build_dataset(cfg, is_train, is_source, load_labels)
    dataset_size = len(dataset)
    split = int(0.9 * dataset_size)

    # 前90%为训练集，后10%为验证集
    train_indices = list(range(0, split))
    val_indices = list(range(split, dataset_size))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=images_per_gpu,
    shuffle=shuffle,        # 训练集可设置为 True 以便打乱数据
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=images_per_gpu,
    shuffle=False,          # 验证集通常不需要打乱
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
    )
    return val_loader
