import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
1、读取数据集，划分数据集
'''
data=pd.read_csv('./diabetes.csv')
# print(data.info())
# print(data.shape) # 768,9
# print(data.head())
# print(data.groupby('Outcome').size())#0:阴性，正常 1：阳性，异常

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

'''
2、模型比较，采用交叉验证分数
'''
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import KFold,cross_val_score

models = []
models.append(("KNN", KNeighborsClassifier(n_neighbors=2)))
models.append(("KNN with weights", KNeighborsClassifier(
    n_neighbors=2, weights="distance")))
models.append(("Radius Neighbors", RadiusNeighborsClassifier(
    n_neighbors=2, radius=500.0)))

result=[]
for name,model in models:
    cv=KFold(n_splits=10)
    cv_result=cross_val_score(model,x,y,cv=cv)
    result.append((name,cv_result))

for name,cv_result in result:
    print('模型名称：{},交叉验证分数：{}'.format(name,cv_result.mean()))

'''
3、模型训练及分析，绘制学习曲线
    通过第二步，选择普通KNN效果好些
'''
#根据分数表现初步判断欠拟合、正常、过拟合
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
train_score=knn.score(x_train,y_train)
test_score=knn.score(x_test,y_test)
print('选择较好的模型测试，训练分数：{},测试分数:{}'.format(train_score,test_score))

from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve

cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
plt.figure()
plot_learning_curve(knn,'Learn Curve for KNN Diabetes',x,y,ylim=(0,1),cv=cv)
plt.show()
#该模型有些欠拟合

'''
4、特征选择及数据可视化
'''
from sklearn.feature_selection import SelectKBest

selector=SelectKBest(k=2)
x_new=selector.fit_transform(x,y)

result=[]
for name,model in models:
    cv=KFold(n_splits=10)
    cv_result=cross_val_score(model,x_new,y,cv=cv)
    result.append((name,cv_result))

for name,cv_result in result:
    print('特征选择模型名称：{},交叉验证分数：{}'.format(name,cv_result.mean()))

plt.figure(figsize=(10, 6))
plt.ylabel("BMI")
plt.xlabel("Glucose")
plt.scatter(x_new[y==0][:,0],x_new[y==0][:,1],s=100,c='g',marker='o')
plt.scatter(x_new[y==1][:,0],x_new[y==1][:,1],s=100,c='r',marker='^')
plt.show()
#该图像中在BMI和Glucose一致的范围内，阴性阳性混合在一起，knn效果不会好的

