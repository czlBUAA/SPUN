import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
class BGnet(nn.Module):
    def __init__(self, model, out_dim):
        super(BGnet,self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Linear(2048, out_dim)
    def forward(self, x):
        return self.resnet(x)