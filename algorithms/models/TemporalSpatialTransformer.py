'''
Date: 2024-01-01
Author: zm
Introduction: Combine CNN and Transformer for spatial-temporal data

This model include the following parts:
    1. TemporalSpatialEmbedding
    2. TemporalSpatialTransformer
'''
import torch
import torch.nn as nn
import os
print(os.getcwd())
from models.IRMSTransformer import IRMSTransformer


class TemporalSpatialEmbedding(nn.Module):
    '''
        使用CNN对时空数据进行处理，精炼空间数据，最后将时空数据转换为包含空间信息的时间序列
    '''
    def __init__(self, input_dim, output_dim, kernel_size):
        super(TemporalSpatialEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
            x: (batch_size, seq_len, bands_num, height, width)
        '''
        batch_size, seq_len, bands_num, height, width = x.shape
        x = x.contiguous().view(batch_size*seq_len, bands_num, height, width)
        x = self.conv(x)
        x = x.view(batch_size, seq_len, bands_num)
        return x

class TemporalSpatialTransformer(nn.Module):
    '''
        使用Transformer对时空数据进行处理，精炼时序数据，最后将时空数据转换为包含时序信息的空间序列
    '''
    def __init__(self,
                    S1_encoder_config: dict,
                    S2_encoder_config: dict,
                    fusion_block_config: dict,
                    num_classes: int,
                    S1_TemporalSpatialEmbedding_config: dict,
                    S2_TemporalSpatialEmbedding_config: dict,
                    as_backbone=False
                    ):
        super(TemporalSpatialTransformer, self).__init__()
        if as_backbone:
            self.out_dim = fusion_block_config["d_model"]*fusion_block_config["seq_length"]
            self.fusion_block_config = fusion_block_config
        self.S1_TemporalSpatialEmbedding = TemporalSpatialEmbedding(S1_TemporalSpatialEmbedding_config["input_dim"],
                                                                  S1_TemporalSpatialEmbedding_config["output_dim"],
                                                                  S1_TemporalSpatialEmbedding_config["kernel_size"])
        self.S2_TemporalSpatialEmbedding = TemporalSpatialEmbedding(S2_TemporalSpatialEmbedding_config["input_dim"],
                                                                  S2_TemporalSpatialEmbedding_config["output_dim"],
                                                                  S2_TemporalSpatialEmbedding_config["kernel_size"])
        self.IRMST = IRMSTransformer(S1_encoder_config,
                                               S2_encoder_config,
                                               fusion_block_config,
                                               num_classes,
                                               as_backbone)

    def forward(self, S1, S2, S2_doy):
        '''
            S1, S2: (batch_size, seq_len, bands_num, height, width)
        '''
        S1 = self.S1_TemporalSpatialEmbedding(S1)  # (batch_size, seq_len, bands_num)
        S2 = self.S2_TemporalSpatialEmbedding(S2)  # (batch_size, seq_len, bands_num)
        S2 = torch.cat([S2, S2_doy], dim=-1) # (batch_size, seq_len, bands_num+1)
        output = self.IRMST(S1, S2)
        return output

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    # test

    model = TemporalSpatialTransformer(
        S1_encoder_config={
            "in_channels": 2,
            "hidden_dim": 64,
            "seq_length": 12,
            "d_model": 64,
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4
        },
        S2_encoder_config={
            "in_channels": 10,
            "seq_length": 100,
            "hidden_dim": 64,
            "time_channels": 1,
            "d_model": 64,
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4
        },
        fusion_block_config={
            "d_model": 128,  # 由于两个特征进行了拼接，所以这里的d_model是两个特征的d_model之和
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4,
            "seq_length": 100
        },
        num_classes=3,
        as_backbone=False,
        S1_TemporalSpatialEmbedding_config={
            "input_dim": 2,
            "output_dim": 2,
            "kernel_size": 5
        },
        S2_TemporalSpatialEmbedding_config={
            "input_dim": 10,
            "output_dim": 10,
            "kernel_size": 5
        }
    )
    S1 = torch.randn(32, 12, 2, 5, 5)
    S2 = torch.randn(32, 100, 10, 5, 5)
    S2_doy = torch.randn(32, 100, 1)
    print(model(S1, S2, S2_doy).shape)
    # torch.Size([2, 5, 4])