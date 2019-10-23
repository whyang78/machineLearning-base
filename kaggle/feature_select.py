import pandas as pd
# 从互联网读取titanic数据。
titanic = pd.read_csv('./Datasets/Titanic/train.csv')

# 分离数据特征与预测目标。
y = titanic['Survived']
X = titanic.drop(['Cabin','Name', 'Survived'], axis = 1)

# 对对缺失数据进行填充。
X['Age'].fillna(X['Age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

# 分割数据，依然采样25%用于测试。
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 类别型特征向量化。 one_hot类型
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 输出处理后特征向量的维度。
print(len(vec.feature_names_))


# 使用决策树模型依靠所有特征进行预测，并作性能评估。
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
dt.score(X_test, y_test)


# 从sklearn导入特征筛选器。
from sklearn.feature_selection import SelectPercentile,chi2
# 筛选前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能。
fs = SelectPercentile(chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)


# 通过交叉验证（下一节将详细介绍）的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化。
from sklearn.model_selection import cross_val_score
import numpy as np

percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = SelectPercentile(chi2, percentile = i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print(results)

# 找到提现最佳性能的特征筛选的百分比。
opt = np.where(results == results.max())[0]
print('Optimal number of features %d' %percentiles[opt])


import matplotlib.pyplot as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()


# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估。
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)