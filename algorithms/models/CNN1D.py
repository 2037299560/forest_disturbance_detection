import torch
import torch.nn as nn



class CNN1D(nn.Module):
    def __init__(self, in_channel, out_channel,
                 dropout=0.6):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.inception_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.inception_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=1), 
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.dropout3 = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(512, out_channel)
        
    
    def forward(self, x):
        # 输入数据为(batch_size, seq_length, in_channels), 需要先转换为(batch_size, in_channels, seq_length)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.pooling(x)
        tmp_conv1 = self.inception_conv1(x)
        tmp_conv2 = self.inception_conv2(x)
        x = torch.cat([tmp_conv1, tmp_conv2], dim=1)  # 在通道维度进行拼接
        x = self.pooling(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    # 生成随机数用于测试网络
    x = torch.randn(128, 100, 11)
    model = CNN1D(in_channel=11, out_channel=10)
    model(x)