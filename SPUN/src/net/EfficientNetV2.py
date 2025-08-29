import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

class EfficientNetV2S(nn.Module):
    def __init__(self, model, out_dim):
        super(EfficientNetV2S, self).__init__()
        self.backbone = model
        # 获取分类器的输入特征维度
        in_features = self.backbone.classifier[1].in_features
        # 替换分类器为回归头，输出指定维度的向量
        self.backbone.classifier = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.backbone(x)
