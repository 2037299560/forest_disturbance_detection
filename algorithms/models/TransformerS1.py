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



class TransformerS1(nn.Module):
    def __init__(self,
                 S1_encoder_config: dict,
                 fusion_block_config: dict,
                 num_classes: int,
                 ):
        super(TransformerS1, self).__init__()
        self.S1_encoder_config = S1_encoder_config
        self.fusion_block_config = fusion_block_config
        self.num_classes = num_classes
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

    def forward(self, S1):
        b1, l1, c1 = S1.shape
        # S1: (batch_size, seq_length, in_channels) 特征提取
        # S1_padding_mask = (S1[:, :, 0] == 0)
        S1 = self.S1_embedding(S1)
        S1 = S1.permute(1, 0, -1)  # 先将x的维度变为(seq_length, batch_size, d_model)
        S1 += self.S1_positional_encoding(S1)  # 加上位置编码
        S1 = self.S1_encoder(S1)  # 计算带掩膜的多头注意力
        S = self.feature_fusion(S1)  # 计算带掩膜的多头注意力
        # 分类
        S = S.permute(1, 0, 2)
        S = S.contiguous().view(b1, -1)
        out = self.fc(S)

        return out

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__:
    irtransformer = TransformerS1(
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
        fusion_block_config={
            "d_model": 64,  # 由于两个特征进行了拼接，所以这里的d_model是两个特征的d_model之和
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4,
            "seq_length": 12
        },
        num_classes=3,
    )
    print(irtransformer(S1=torch.randn(128, 12, 2)).shape
          )
