import torch.nn as nn
import torch

# 用于变化检测任务
class TemporalSpatialTransformerBaseContrastive(nn.Module):
    def __init__(self, backbone, ts_length=100, num_classes=3):
        super(TemporalSpatialTransformerBaseContrastive, self).__init__()
        self.backbone = backbone
        self.backbone_dim = backbone.out_dim
        self.num_classes = num_classes
        self.ts_length = ts_length
        fusion_block_config = backbone.fusion_block_config

        self.classification_head = nn.Sequential(
            nn.Linear(fusion_block_config["d_model"]*fusion_block_config["seq_length"], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(fusion_block_config["dropout"]),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(fusion_block_config["dropout"]),
            nn.Linear(256, num_classes)
        )


    def forward(self, s1, s2, doy):
        x = self.backbone(s1, s2, doy)  # (batch size, ts_length, )
        output = self.classification_head(x)  # (batch size, change_type_num)
        return output

    def load_backbone_weight(self, contrastive_model, path):
        contrastive_model.load_state_dict(torch.load(path))
        self.backbone.load_state_dict(contrastive_model.backbone.state_dict())
        print("Load weight from {}".format(path))
