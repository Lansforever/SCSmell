import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer

# 定义一个简单的线性层网络，将输入维度从 768 降到 16
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(768, 16)
        self.linear2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# 加载预训练的BERT模型
model = SentenceTransformer('bert-base-nli-mean-tokens')

# 读取CSV文件
df = pd.read_csv(r'G:\项目代码文本\总文本.csv')

# 将字符串转换成向量
vectors = model.encode(df['Content'].values)

# 构建神经网络
net = LinearNet()

# 将向量添加到DataFrame中
df = pd.concat([df.drop('Content', axis=1), pd.DataFrame(vectors)], axis=1)

# 将DataFrame转换成PyTorch张量
x = torch.tensor([item for item in df.values]).float()

# 计算投影后的结果
projection = net(x)

# 保存转换后的DataFrame
df_projection = pd.DataFrame(projection.detach().numpy(), columns=['dim1', 'dim2', 'dim3'])
df_projection.to_csv(r'G:\项目代码文本\总文本-bert-projection.csv', index=False)