import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

def load_csv(path):
    # 加载CSV文件，从第6列到倒数第2列
    df = pd.read_csv(path)
    df = df.round(5)
    df = df.fillna(0)
    features = df.iloc[:, 5:-1].values
    return features

class ContrastiveDataset(Dataset):
    def __init__(self, doh_data, nondoh_data):
        self.doh_data = doh_data
        # 3. 欠采样，减少nondoh样本数量
        if len(nondoh_data) > len(doh_data):
            nondoh_data = nondoh_data[:len(doh_data)]  # 截取nondoh_data的前len(doh_data)个样本
        self.nondoh_data = nondoh_data

    def __len__(self):
        # 数据集的长度是正负样本对的数量之和
        return len(self.doh_data) + len(self.nondoh_data)

    def __getitem__(self, idx):
        if idx < len(self.doh_data):
            # 正样本对
            anchor = self.doh_data[idx]
            positive = self.doh_data[(idx + 1) % len(self.doh_data)]  # 从DoH中选择一个正样本
            label = 1  # 正样本对标签为1
        else:
            # 负样本对
            anchor = self.doh_data[idx % len(self.doh_data)]  # 从DoH中选择一个样本
            positive = self.nondoh_data[idx % len(self.nondoh_data)]  # 从Non-DoH中选择一个负样本
            label = 0  # 负样本对标签为0

        # 将anchor和positive转换为torch张量
        return torch.tensor(anchor, dtype=torch.float32), torch.tensor(positive, dtype=torch.float32), label


if __name__ == '__main__':
    print(torch.cuda.is_available())
