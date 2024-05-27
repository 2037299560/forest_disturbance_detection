import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于对比学习的模型
class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = backbone.out_dim
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim // 2),
                nn.ReLU(),
                nn.Linear(self.backbone_dim // 2, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, s1, s2, doy):
        x = self.backbone(s1, s2, doy)
        features = self.contrastive_head(x.view(-1, self.backbone_dim))
        # features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features
