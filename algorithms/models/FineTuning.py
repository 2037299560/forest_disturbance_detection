"""
创建于：2023/06/26
功能: 使用预训练好的模型，结合有监督进行干扰检测模型的微调
"""

import os
import torch
import torch.nn as nn


class FineTuning(nn.Module):
    def __init__(self, backbone, backbone_out_dims, num_classes):
        super(FineTuning, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_out_dims, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        # 冻结骨干网络参数
        for param in self.backbone.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone.encode(x, encoding_window="full_series")
        x.to(torch.device('cuda:0'))
        x = self.classifier(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
