# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 读取数据
data = pd.read_csv(r'G:\项目度量与文本结合-Smote\BrainClass.csv')

# 划分训练集和测试集
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 第一层模型
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_preds = np.zeros((X_train.shape[0],))
lr_preds = np.zeros((X_train.shape[0],))
svm_preds = np.zeros((X_train.shape[0],))

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    # 随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_fold, y_train_fold)
    rf_preds[val_index] = rf.predict(X_val_fold)
    # 逻辑回归模型
    lr = LogisticRegression(penalty=12,tol=0.0001,max_iter=100)
    lr.fit(X_train_fold, y_train_fold)
    lr_preds[val_index] = lr.predict(X_val_fold)

    # 支持向量机模型
    svm = SVC(degree=3,tlo=0.001,cache_size=200)
    svm.fit(X_train_fold, y_train_fold)
    svm_preds[val_index] = svm.predict(X_val_fold)

# 第二层模型
X_train_second = pd.DataFrame({'rf_preds': rf_preds, 'lr_preds': lr_preds, 'svm_preds': svm_preds})
X_test_second = pd.DataFrame({'rf_preds': rf.predict(X_test), 'lr_preds': lr.predict(X_test), 'svm_preds': svm.predict(X_test)})

# 随机森林模型
rf_second = RandomForestClassifier(n_estimators=100, random_state=42)
rf_second.fit(X_train_second, y_train)
rf_second_preds = rf_second.predict(X_test_second)

# 第三层模型
final_preds = rf_second_preds

# 评估指标
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('recall:', recall)
print('f1:', f1)