''' build a model based transformer '''
import torch.nn as nn
import torch

class DisturbanceTypeBasedTransformer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 seq_length: int,
                 d_model: int,
                 n_layers: int,
                 d_feedforward: int,
                 n_head: int,
                 dropout: float,
                 in_channels: int = 11
                 ):
        super(DisturbanceTypeBasedTransformer, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.n_layers = n_layers
        self.d_feedforward = d_feedforward
        self.num_classes = num_classes
        self.n_head = n_head
        self.dropout = dropout
        self.in_channels = in_channels

        self.embedding = nn.Linear(in_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_length)   # (batch_size, seq_length, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_feedforward,
                dropout=dropout,
                activation='relu'
            ),
            num_layers=n_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(seq_length * d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        # self.fc = nn.Linear(seq_length * d_model, num_classes)

    def forward(self, x):
        # 创建填充掩码
        padding_mask = (x[:, :, 0] == 0)  # 将填充位置标记为True，非填充位置标记为False
        x = self.embedding(x)
        x = x.permute(1,0,-1)   # 先将x的维度变为(seq_length, batch_size, d_model)
        x += self.positional_encoding(x)  # 加上位置编码
        x = self.transformer(x, src_key_padding_mask=padding_mask)   # 计算带掩膜的多头注意力
        x = x.permute(1,0,-1)   # 再将x的维度变回来(batch_size, seq_length, d_model)
        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, seq_length):
      super(PositionalEncoding,self).__init__()
      pe=torch.zeros(seq_length,d_model)
      position=torch.arange(0.,seq_length).unsqueeze(1)
      div_term=torch.exp(torch.arange(0.,d_model,2)*-(torch.log(torch.tensor(10000.))/d_model))
      pe[:,0::2]=torch.sin(position*div_term)
      pe[:,1::2]=torch.cos(position*div_term)
      pe=pe.unsqueeze(0).transpose(0,-2)
      self.register_buffer('pe',pe)

    def forward(self,x):
      return x+self.pe

if __name__ == "__main__":
    model = DisturbanceTypeBasedTransformer(
        num_classes=3,
        seq_length=100,
        d_model=256,
        n_layers=2,
        d_feedforward=256,
        n_head=8,
        dropout=0.5,
        in_channels=11
    )
    print(model)
    print(model(torch.randn(4, 100, 11)).shape)
    