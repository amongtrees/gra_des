import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_dim=128, num_layers=6, nhead=8):
        super(TransformerModel, self).__init__()
        # 输入映射层
        self.embedding = nn.Linear(input_size, hidden_dim)

        # Transformer 编码器层
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                                    dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层到最终嵌入向量
        self.norm = nn.LayerNorm(hidden_dim)  # LayerNorm 归一化

    def forward(self, x):
        # 先通过线性层将输入转换为隐藏维度
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Transformer要求输入维度为 (seq_len, batch_size, features)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 对每个样本的所有词向量求均值
        x = self.fc(x)
        x = self.norm(x)  # 对最终输出使用LayerNorm
        return x
