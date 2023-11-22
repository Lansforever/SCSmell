import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

# 读取 CSV 文件
df = pd.read_csv(r'G:\项目度量信息\FeatureEnvy.csv')

# 分离特征和目标
X = df.drop(columns=['NAME', 'label'])
y = df['label']

# 构建线性回归模型
model = LinearRegression()

# 构建 REFVC 特征选择模型
rfecv = RFECV(estimator=model, cv=5, step=1)

# 运行 REFVC 特征选择算法
rfecv.fit(X, y)

# 保留重要的特征和原始方法名列、目标列
selected_features = X.columns[rfecv.support_][:10]  # 取前10个特征
df_selected_features = df.loc[:, ['label'] + list(selected_features)]

# 保存结果到 CSV 文件
df_selected_features.to_csv(r'G:\项目度量-REFCV\FeatureEnvy-REFCV.csv', index=False)