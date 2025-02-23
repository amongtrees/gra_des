import torch.nn.functional as F
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, label):
        # 计算欧氏距离
        euclidean_distance = F.pairwise_distance(anchor, positive, keepdim=True)

        # 对比损失：对于正样本，尽量最小化距离；对于负样本，尽量增大距离
        loss = 0.5 * (label.float() * torch.pow(euclidean_distance, 2) +
                      (1 - label.float()) * torch.pow(F.relu(self.margin - euclidean_distance), 2))
        return loss.mean()
