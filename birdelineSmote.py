import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE

# 1. 导入必要的库和工具

# 2. 读取原始CSV文件并将其转换为pandas DataFrame
df = pd.read_csv(r'G:\项目度量与文本结合\BrainMethod.csv')
df = df.iloc[:, 1:]
df.fillna(value=0)  # 用 0 填充所有缺失值
df.fillna(method='ffill')  # 用上一行数据填充所有缺失值
# 3. 将数据集分为正负两类样本，计算它们的数量并确定要生成的复制倍数以达到所需的正负样本比例。
positive_samples = df[df['label'] == 1] # 正样本
negative_samples = df[df['label'] == 0] # 负样本
n_positive = len(positive_samples)
n_negative = len(negative_samples)
r = 0.5
n_negative_new = int(n_positive * r * 1.0) # 目标负样本数
n_positive_new = n_positive  # 目标正样本数

# 4. 对于少数派样本，使用Birdeline-SMOTE将其随机复制并生成新的样本。
if n_negative_new > n_negative:
  birdelineSmote = BorderlineSMOTE(sampling_strategy={0:n_negative_new, 1:n_positive_new})
  x_res, y_res = birdelineSmote.fit_resample(df.iloc[:, :-1], df.iloc[:, -1])
else:
  birdelineSmote = BorderlineSMOTE()
  x_res, y_res = birdelineSmote.fit_resample(df.iloc[:, :-1], df.iloc[:, -1])

# 5. 将新生成的样本列表合并到原始数据集中。
df_new = pd.concat([pd.DataFrame(x_res), pd.DataFrame(y_res, columns=['label'])], axis=1)

# 6. 将带有平衡样本的DataFrame保存为新的CSV文件。
df_new.to_csv(r'G:\项目度量与文本结合-Smote\BrainMethod.csv', index=False)