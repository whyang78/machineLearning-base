import matplotlib.pyplot as plt
import numpy as np

'''
1、数据采集及特征提取
'''
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
# print(x.shape)
# print(y[y==1].shape[0],y[y==0].shape[0])

'''
2、模型训练
'''
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=78)
# clf=LogisticRegression()
# clf.fit(x_train,y_train)
# train_score=clf.score(x_train,y_train)
# test_score=clf.score(x_test,y_test)
# print('训练分数:{:.2f},测试分数:{:.2f}'.format(train_score,test_score))
#
# y_pred=clf.predict(x_test)
# print('预测结果:{}/{}'.format(np.equal(y_pred,y_test).sum(),y_test.shape[0]))
#
# #结果置信度低于90%
# y_pred_prob=clf.predict_proba(x_test)
# result=y_pred_prob[y_pred_prob[:,0]>0.1,:]
# print(result[result[:,1]>0.1,:])
#
# '''
# 3、模型优化 多项式处理 正则化
# '''
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
#
# def polynomial_model(degree=1, **kwarg):
#     polynomial_feature=PolynomialFeatures(degree=degree)
#     logistic_regression=LogisticRegression(**kwarg)
#     model=Pipeline([('polynomial_feature',polynomial_feature),
#                     ('logistic_regression',logistic_regression)])
#     return model
#
# from learning_curve import plot_learning_curve
# from sklearn.model_selection import ShuffleSplit
#
# cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
# degrees=[1,2]
# penalty=['l1','l2']
# title='learning curve with degree:{0},penalty:{1}'
# plt.figure(figsize=(8,8))
# index=1
# for i in range(len(degrees)):
#     for j in range(len(penalty)):
#         plt.subplot(len(degrees),len(penalty),index)
#         model=polynomial_model(degrees[i],penalty=penalty[j])
#         plot_learning_curve(estimator=model,X=x,y=y,ylim=[0.89,1.01],
#                             title=title.format(degrees[i],penalty[j]),
#                             cv=cv
#                             )
#         index+=1
# plt.show()