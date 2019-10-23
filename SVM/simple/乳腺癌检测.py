import numpy as np
import matplotlib.pyplot as plt

'''
在此实验中，我们可以发现高斯核函数具有过拟合现象，且gamma很小时性能也低于二阶多项式核函数，
所以针对这种过拟合现象，可以考虑简化模型，最后采用多项式特征预处理的线性分类器，即增加数据
特征，同时简化模型。
'''

'''
1、加载数据
'''
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
x=cancer.data
y=cancer.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=78)

'''
2、使用高斯核函数进行分类
'''
from sklearn.svm import SVC
clf=SVC(kernel='rbf',gamma=0.01)
clf.fit(x_train,y_train)
train_score=clf.score(x_train,y_train)
test_score=clf.score(x_test,y_test)
print('gamma=0.01时训练分数：{}，测试分数：{}'.format(train_score,test_score))
#过拟合

from sklearn.model_selection import GridSearchCV
from param_curve import plot_curve

gammas=np.linspace(0,0.0003,30)
clf=GridSearchCV(SVC(),{'gamma':gammas},cv=5)
clf.fit(x,y)
print('最优参数：{}，测试分数：{}'.format(clf.best_params_,clf.best_score_))
# 最优参数：{'gamma': 0.00011379310344827585}，测试分数：0.9367311072056239
plt.figure(figsize=(10,4))
plot_curve(gammas,cv_result=clf.cv_results_,xlabel='gammas')

from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve

cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
clf=SVC(gamma=0.00011379310344827585)
plt.figure(figsize=(10,4))
plot_learning_curve(clf,'learning curve for svc',x,y,ylim=[0.89,1.01],cv=cv)
plt.show()

'''
3、使用多项式核函数进行分类
'''
clf=SVC(kernel='poly',degree=2)
clf.fit(x_train,y_train)
train_score=clf.score(x_train,y_train)
test_score=clf.score(x_test,y_test)
print('采用多项式核函数时训练分数：{}，测试分数：{}'.format(train_score,test_score))
# 挺不错的

from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve

degrees=[1,2]
title='learning curve with degree:{}'
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
plt.figure(figsize=(10,4))
for i in range(len(degrees)):
    plt.subplot(1,2,i+1)
    clf=SVC(kernel='poly',degree=degrees[i])
    plot_learning_curve(clf,title.format(degrees[i]),x,y,ylim=[0.89,1.01],cv=cv)
plt.show()

'''
4、使用多项式线性SVM
'''
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def model(degree=2,**kwargs):
    polynomial_feature=PolynomialFeatures(degree=degree,include_bias=False)
    minmaxScale=MinMaxScaler()
    linearSvc=LinearSVC(**kwargs)
    pipeline=Pipeline([('polynomial_feature',polynomial_feature),
                       ('minmaxScale',minmaxScale),
                       ('linearSvc',linearSvc)])
    return pipeline

clf=model(penalty='l1',dual=False)
clf.fit(x_train,y_train)
train_score=clf.score(x_train,y_train)
test_score=clf.score(x_test,y_test)
print('LinearSVC时训练分数：{}，测试分数：{}'.format(train_score,test_score))

from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve

penaltys=['l1','l2']
title='learning curve with penalty:{}'
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
plt.figure(figsize=(10,4))
for i in range(len(penaltys)):
    plt.subplot(1,2,i+1)
    clf = model(penalty=penaltys[i], dual=False)
    plot_learning_curve(clf,title.format(penaltys[i]),x,y,ylim=[0.89,1.01],cv=cv)
plt.show()