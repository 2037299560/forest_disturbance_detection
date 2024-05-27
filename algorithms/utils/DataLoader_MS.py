from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os


class MSTimeSeriesData(Dataset):
    def __init__(self, S1_data_path, S1_input_bands, S2_data_path, S2_input_bands):
        '''
        data_path: the dir of multi-band csv data
        a examples: ''data/train/''
        '''
        # read multi-band csv data from data_path
        self.S1_data = None
        self.S2_data = None
        self.labels = None

        # 读取Sentinel-1数据
        S1_data = []
        for idx, band in enumerate(S1_input_bands):
            tmp = pd.read_csv(os.path.join(S1_data_path, f"{band}.csv"), header=None, index_col=None).to_numpy()
            S1_data.append(tmp[:, 1:].reshape(1, tmp.shape[0], -1))
        S1_data = np.concatenate(S1_data, axis=0)
        # 去除nan
        S1_data[np.isnan(S1_data)] = 0
        # 去除缺失值，指的是遥感意义上的缺失值，-32768
        S1_data[S1_data < -10000] = 0
        self.S1_data = S1_data

        # 读取Sentinel-2数据
        S2_data = []
        for idx, band in enumerate(S2_input_bands):
            tmp = pd.read_csv(os.path.join(S2_data_path, f"{band}.csv"), header=None, index_col=None).to_numpy()
            ### 临时实验代码：减少一部分无扰动样本，代码为0 ###
            # cnt = tmp[tmp[:, 0] == 1, :]
            # cnt2 = tmp[tmp[:, 0] == 0, :][::10, :]
            # if idx == 0:
            #     print(cnt.shape, cnt2.shape)
            # tmp = np.concatenate([cnt, cnt2], axis=0)
            ### ###
            tmp1 = tmp[:, 1:].reshape(1, tmp.shape[0], -1)
            if band != "DOY":
                tmp1[np.isnan(tmp1)] = 0
                tmp1[tmp1 < -10000] = 0
                tmp1 = tmp1 / 10000.0
            else:
                tmp1 = tmp1 / 365.0
            S2_data.append(tmp1)
            if idx == 0:
                self.labels = tmp[::, 0]
        self.S2_data = np.concatenate(S2_data, axis=0)



    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # 提取Sentinel-1数据
        S1 = torch.from_numpy(self.S1_data[:, index, :]).float()
        S1 = S1.transpose(0, 1)  # (seq_length, in_channels)

        # 提取Sentinel-2数据
        S2 = torch.from_numpy(self.S2_data[:, index, :]).float()
        S2 = S2.transpose(0, 1)  # (seq_length, in_channels)

        # 提取标签
        label = torch.tensor(self.labels[index]).long() 
        return (S1, S2), label

# 测试一下是否可以正常加载数据
if __name__ == "__main__":
    data_loader = DataLoader(dataset=MSTimeSeriesData(S1_data_path="./data/supervised/original_bands/train/", S1_input_bands=["VV", "VH"],
                                                      S2_data_path="./data/supervised/original_bands/train/", S2_input_bands= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']),
                             batch_size=128,
                             shuffle=True)
    for batch_idx, (data, targets) in enumerate(data_loader):
        print(data[0].shape, data[1].shape)
        print(targets.shape)
        break
