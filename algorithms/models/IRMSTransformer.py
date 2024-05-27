"""
date: 2023.08.12
author: ZM
introduction: the Transformer model for irregular time series

This model include the following parts:
    1. Tuple embedding
    2. Transformer encoder
    3. Feature fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0., seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(torch.log(torch.tensor(10000.)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, -2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class TupleEmbddingModule(nn.Module):
    """
    Tuple embedding module
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(TupleEmbddingModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            x: (batch_size, seq_len, output_dim)
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class IRMSTransformer(nn.Module):
    def __init__(self,
                 S1_encoder_config: dict,
                 S2_encoder_config: dict,
                 fusion_block_config: dict,
                 num_classes: int,
                 as_backbone=False
                 ):
        super(IRMSTransformer, self).__init__()
        self.S1_encoder_config = S1_encoder_config
        self.S2_encoder_config = S2_encoder_config
        self.fusion_block_config = fusion_block_config
        self.num_classes = num_classes
        self.as_backbone = as_backbone
        if self.as_backbone:
            self.out_dim = fusion_block_config["d_model"]*fusion_block_config["seq_length"]

        # Tuple embedding module, S1 data not need to be embedded
        self.S2_time_embedding = TupleEmbddingModule(input_dim=self.S2_encoder_config["time_channels"],
                                                output_dim=self.S2_encoder_config["d_model"],
                                                hidden_dim=self.S2_encoder_config["hidden_dim"])
        self.S2_value_embedding = TupleEmbddingModule(input_dim=self.S2_encoder_config["in_channels"],
                                                output_dim=self.S2_encoder_config["d_model"],
                                                hidden_dim=self.S2_encoder_config["hidden_dim"])

        # Feature extracted module
        self.S1_embedding = nn.Linear(S1_encoder_config["in_channels"], S1_encoder_config["d_model"])
        self.S1_positional_encoding = PositionalEncoding(S1_encoder_config["d_model"], S1_encoder_config["seq_length"])  # (batch_size, seq_length, d_model)
        self.S1_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=S1_encoder_config["d_model"],
                nhead=S1_encoder_config["n_head"],
                dim_feedforward=S1_encoder_config["d_feedforward"],
                dropout=S1_encoder_config["dropout"],
                activation='relu'
            ),
            num_layers=S1_encoder_config["n_layers"]
        )
        self.S2_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=S2_encoder_config["d_model"],
                nhead=S2_encoder_config["n_head"],
                dim_feedforward=S2_encoder_config["d_feedforward"],
                dropout=S2_encoder_config["dropout"],
                activation='relu'
            ),
            num_layers=S2_encoder_config["n_layers"]
        )
        self.S1_embedding_before_fusion = nn.Linear(S1_encoder_config["seq_length"], fusion_block_config["seq_length"])
        # Feature fusion module
        self.feature_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_block_config["d_model"],
                nhead=fusion_block_config["n_head"],
                dim_feedforward=fusion_block_config["d_feedforward"],
                dropout=fusion_block_config["dropout"],
                activation='relu'
            ),
            num_layers=fusion_block_config["n_layers"]
        )

        # Classifier
        self.fc = nn.Sequential(
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

    def forward(self, S1, S2):
        b1, l1, c1 = S1.shape
        b2, l2, c2 = S2.shape
        # S1: (batch_size, seq_length, in_channels) 特征提取
        # S1_padding_mask = (S1[:, :, 0] == 0)
        S1 = self.S1_embedding(S1)
        S1 = S1.permute(1, 0, -1)  # 先将x的维度变为(seq_length, batch_size, d_model)
        S1 += self.S1_positional_encoding(S1)  # 加上位置编码
        S1 = self.S1_encoder(S1)  # 计算带掩膜的多头注意力

        # S2: (batch_size, seq_length, in_channels) 特征提取
        S2_time = self.S2_time_embedding(S2[:, :, -1].view(b2, l2, 1))  # (batch_size, seq_length, band_num) 最后一个波段为DOY
        S2_value = self.S2_value_embedding(S2[:, :, :-1])  # (batch_size, seq_length, band_num) 前面的波段为S2的波段
        S2 = S2_time + S2_value
        S2 = S2.permute(1, 0, -1)  # 先将x的维度变为(seq_length, batch_size, d_model)
        S2 = self.S2_encoder(S2)  # 计算带掩膜的多头注意力

        # 特征融合前先将S1的形状变成跟S2一样：(100, batch_size, d_model)
        S1 = S1.permute(1, 2, 0)
        S1 = self.S1_embedding_before_fusion(S1)
        S1 = S1.permute(2, 0, 1)

        # 特征融合
        S = torch.cat((S1, S2), dim=2)
        # S_padding_mask = (S[:, :, 0] == 0)  # 将填充位置标记为True，非填充位置标记为False
        S = self.feature_fusion(S)  # 计算带掩膜的多头注意力

        # 分类
        if self.as_backbone:
            S = S.permute(1, 0, 2)
            out = S.contiguous().view(b1, -1)
        else:
            S = S.permute(1, 0, 2)
            S = S.contiguous().view(b1, -1)
            out = self.fc(S)

        return out

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__:
    irtransformer = IRMSTransformer(
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
        as_backbone=True
    )
    print(irtransformer(S1=torch.randn(128, 12, 2), S2=torch.randn(128, 100, 11)).shape
          )
