import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
df = pd.read_csv(r'G:\项目度量与文本结合\FeatureEnvyaaa.csv')

# Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardize the data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Dimensionality reduction using LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

# Create a new dataframe with reduced dimensionality and labels
df_lda = pd.concat([pd.DataFrame(data=X_lda), pd.DataFrame(data=y)], axis=1)

# Save the reduced data to a new csv file
df_lda.to_csv(r"G:\项目代码文本\FeatureEnvy-LDA.csv", index=False, header=False)

