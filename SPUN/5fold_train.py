import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.datasets.build import build_dataset
from src.datasets.Park2019KRNDataset import Park2019KRNDataset
from config import cfg
def build_dataset(cfg, is_train=True, is_source=True, load_labels=True):


    dataset = Park2019KRNDataset(cfg, None, is_train, is_source, load_labels)

    return dataset

# dataset_size 假设是 dataset 的长度
dataset = build_dataset(cfg, True, True, True)
dataset_size = 2791

# 生成所有样本的索引
all_indices = np.arange(dataset_size)

# 设置 5 折交叉验证，打乱顺序
kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds = []
for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
    folds.append({
        "fold": fold,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist()
    })

# 转为 DataFrame，每一行是一个 fold
df = pd.DataFrame(folds)

# 保存到 csv
df.to_csv("cv_indices.csv", index=False)

print("五折交叉验证索引已保存到 cv_indices.csv")
