''' build a model based transformer '''
import torch.nn as nn
import torch


class MSTransformer(nn.Module):
    # 目前包括两个Encoder，分别用来处理Sentinel-1和Sentinel-2数据，后面跟一个融合器（也是Transformer架构），用于时空特征融合
    def __init__(self,
                    S1_encoder_config: dict,
                    S2_encoder_config: dict,
                    fusion_block_config: dict,
                    num_classes: int = 3,
                 ):
        super(MSTransformer, self).__init__()
        #  Sentinel-1 编码器
        self.S1_embedding = nn.Linear(S1_encoder_config["in_channels"], S1_encoder_config["d_model"])
        self.S1_positional_encoding = PositionalEncoding(S1_encoder_config["d_model"], S1_encoder_config["seq_length"])  # (batch_size, seq_length, d_model)
        self.S1_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=S1_encoder_config["d_model"],
                nhead=S1_encoder_config["n_head"],
                dim_feedforward=S1_encoder_config["d_feedforward"],
                dropout=S1_encoder_config["dropout"],
                activation='relu'
            ),
            num_layers=S1_encoder_config["n_layers"]
        )

        #  Sentinel-2 编码器
        self.S2_embedding = nn.Linear(S2_encoder_config["in_channels"], S2_encoder_config["d_model"])
        self.S2_positional_encoding = PositionalEncoding(S2_encoder_config["d_model"], S2_encoder_config["seq_length"])  # (batch_size, seq_length, d_model)
        self.S2_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=S2_encoder_config["d_model"],
                nhead=S2_encoder_config["n_head"],
                dim_feedforward=S2_encoder_config["d_feedforward"],
                dropout=S2_encoder_config["dropout"],
                activation='relu'
            ),
            num_layers=S2_encoder_config["n_layers"]
        )

        # 嵌入层，特征融合前将S1的形状变成跟S2一样：(100, batch_size, d_model)
        self.S1_embedding_before_fusion = nn.Linear(S1_encoder_config["seq_length"], fusion_block_config["seq_length"])

        # 特征融合器
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

        # 分类器
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

    def forward(self, S1, S2, doy_pe=False, doy=None):
        # S1: (batch_size, seq_length, in_channels) 特征提取
        S1_padding_mask = (S1[:, :, 0] == 0)  # 将填充位置标记为True，非填充位置标记为False
        S1 = self.S1_embedding(S1)
        S1 = S1.permute(1, 0, -1)  # 先将x的维度变为(seq_length, batch_size, d_model)
        S1 = self.S1_positional_encoding(S1)  # 加上位置编码
        S1 = self.S1_encoder(S1, src_key_padding_mask=S1_padding_mask)  # 计算带掩膜的多头注意力

        # S2: (batch_size, seq_length, in_channels) 特征提取
        S2_padding_mask = (S2[:, :, 0] == 0)  # 将填充位置标记为True，非填充位置标记为False
        S2 = self.S2_embedding(S2)
        S2 = S2.permute(1, 0, -1)  # 先将x的维度变为(seq_length, batch_size, d_model)
        if doy_pe:
            S2 += doy.permute(1, 0, -1)       # 使用doy数据作为位置编码
        else:
            S2 = self.S2_positional_encoding(S2)  # 加上位置编码
        S2 = self.S2_encoder(S2, src_key_padding_mask=S2_padding_mask)  # 计算带掩膜的多头注意力

        # 特征融合前先将S1的形状变成跟S2一样：(100, batch_size, d_model)
        S1 = S1.permute(1, 2, 0)
        S1 = self.S1_embedding_before_fusion(S1)
        S1 = S1.permute(2, 0, 1)

        # 特征融合
        S = torch.cat((S1, S2), dim=2)
        # S_padding_mask = (S[:, :, 0] == 0)  # 将填充位置标记为True，非填充位置标记为False
        S = self.feature_fusion(S)  # 计算带掩膜的多头注意力

        # 分类
        S = S.permute(1, 0, 2)  # 再将x的维度变回来(batch_size, seq_length, d_model)
        S = S.reshape(S.shape[0], -1)
        out = self.fc(S)
        return out

    def load(self, path):
        self.load_state_dict(torch.load(path))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0., seq_length).unsqueeze(1)   # (seq_length, 1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(torch.log(torch.tensor(10000.)) / d_model))    # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, -2)  # (seq_length, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


if __name__ == "__main__":
    model = MSTransformer(
        S1_encoder_config={
            "in_channels": 2,
            "seq_length": 12,
            "d_model": 64,
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.3
        },
        S2_encoder_config={
            "in_channels": 10,
            "seq_length": 100,
            "d_model": 64,
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.3
        },
        fusion_block_config={
            "d_model": 128,  # 由于两个特征进行了拼接，所以这里的d_model是两个特征的d_model之和
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.3,
            "seq_length": 100
        },
        num_classes=3
    )
    print(model)
    print(model(torch.randn(4, 12, 2), torch.randn(4, 100, 10)).shape)
