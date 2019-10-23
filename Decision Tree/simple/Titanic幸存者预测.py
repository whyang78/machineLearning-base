import pandas as pd
import numpy as np

'''
1、数据分析，预处理
'''
train_dataset='./titanic/train.csv'
test_dataset='./titanic/test.csv'
data=pd.read_csv(train_dataset,index_col=0)
# for col in data.columns:
#     print('*'*15)
#     print(col)
#     print(data[col].value_counts())

#print(data.isna().sum())#统计每列的缺失值个数
#去除一些无关特征 前两个属于无关特征，最后一个有关但是缺失值太多，故舍弃
data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
#处理性别数据
data['Sex']=(data['Sex']=='male').astype('int')
#处理乘船港口数据
labels=data['Embarked'].unique().tolist()
data['Embarked']=data['Embarked'].apply(lambda x:labels.index(x))
#method1 填充缺失值 处理到此步时，只剩下age有缺失值，177/891
data.fillna(0,inplace=True)
#method2 删除年龄中包含缺失值的行
# data.dropna(subset=['Age'],inplace=True)

'''
2、模型训练
'''
#数据集
from sklearn.model_selection import train_test_split
y=data['Survived'].values
x=data.drop(['Survived'],axis=1).values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#训练
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.externals.six import StringIO
import pydotplus

# clf=DecisionTreeClassifier()
# clf.fit(x_train,y_train)
# train_score=clf.score(x_train,y_train)
# test_score=clf.score(x_test,y_test)
# # print("训练分数：{},测试分数:{}".format(train_score,test_score))
# #训练分数不错，但是测试分数有些低，所以明显存在过拟合现象，
# #观察决策树，也发现层数有些深
#
# dot_data=StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 feature_names=data.drop(['Survived'],axis=1).keys(),
#                 filled=True,rounded=True,
#                 special_characters=True
#                 )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("titanic_tree_1.pdf")
#
# '''
# 3、优化模型参数 模型参数选择器
# '''
# #绘图函数
# import matplotlib.pyplot as plt
# #针对当某种参数不断变化时，由于划分数据集的随机性，使得结果不同，故采取多次划分，取平均值的做法
# #在单一参数变化中可以采用此种绘图方法
# def plot_curve(param_sizes,cv_result,xlabel):
#     train_score_mean=cv_result['mean_train_score']
#     train_score_std=cv_result['std_train_score']
#     test_score_mean=cv_result['mean_test_score']
#     test_score_std=cv_result['std_test_score']
#
#     plt.figure()
#     plt.title('parameters turning')
#     plt.xlabel(xlabel)
#     plt.ylabel('score')
#     plt.grid()
#
#     plt.fill_between(param_sizes,train_score_mean-train_score_std,
#                      train_score_mean+train_score_std,color='r',alpha=0.2)
#     plt.fill_between(param_sizes,test_score_mean-test_score_std,
#                      test_score_mean+test_score_std,color='g',alpha=0.2)
#     plt.plot(param_sizes,train_score_mean,'.--',color='r',label='train_scores')
#     plt.plot(param_sizes,test_score_mean,'.-',color='g',label='Cross_validation_scores')
#     plt.legend(loc='best')
#
#     plt.show()
#
#
# from sklearn.model_selection import GridSearchCV
# #单一参数选择
# thresholds = np.linspace(0, 0.01, 50)
# clf=GridSearchCV(DecisionTreeClassifier(),{'min_impurity_decrease':thresholds},
#                  cv=5,return_train_score=True)
# clf.fit(x,y)
# print('best_param:{},best_score:{}'.format(clf.best_params_,clf.best_score_))
# plot_curve(thresholds,clf.cv_results_,xlabel='gini thresholds')
#
# #多组参数选择
# entropy_thresholds = np.linspace(0, 0.01, 50)
# gini_thresholds = np.linspace(0, 0.005, 50)
#
# #在列表中某个字典中选择一个最优的，一个字典中可以包含多种参数
# param_grid = [{'criterion': ['entropy'],
#                'min_impurity_decrease': entropy_thresholds},
#               {'criterion': ['gini'],
#                'min_impurity_decrease': gini_thresholds},
#               {'max_depth': range(2, 10)},
#               {'min_samples_split': range(2, 30, 2)}]
#
# clf=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5,return_train_score=True)
# clf.fit(x,y)
# print('best_param:{},best_score:{}'.format(clf.best_params_,clf.best_score_))
# #最佳方案 'criterion': 'entropy', 'min_impurity_decrease': 0.002857142857142857

'''
4、根据最佳方案，测试一下数据，并绘制决策树
'''
# clf=DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=0.002857142857142857)
# clf.fit(x_train,y_train)
# train_score=clf.score(x_train,y_train)
# test_score=clf.score(x_test,y_test)
# print("优秀模型训练分数：{},测试分数:{}".format(train_score,test_score))
#
# dot_data=StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 feature_names=data.drop(['Survived'],axis=1).keys(),
#                 filled=True,rounded=True,
#                 special_characters=True
#                 )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("titanic_tree_2.pdf")

'''
5、预测测试集结果
'''
test_data=pd.read_csv(test_dataset,index_col=0)
original_test_data=test_data.copy()

test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
#处理性别数据
test_data['Sex']=(test_data['Sex']=='male').astype('int')
#处理乘船港口数据
labels=test_data['Embarked'].unique().tolist()
test_data['Embarked']=test_data['Embarked'].apply(lambda x:labels.index(x))
test_data.fillna(0,inplace=True)

test_vec=test_data.values
clf=DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=0.002857142857142857)
clf.fit(x,y)
result=clf.predict(test_vec)

#将幸存预测结果添加到df中，并保存为csv
# original_test_data['Survived']=result
# original_test_data.to_csv('./在最后一列添加幸存预测情况.csv')

columns=original_test_data.columns.tolist()
columns.insert(0,'Survived')
original_test_data=original_test_data.reindex(columns=columns)
original_test_data['Survived']=result
original_test_data.to_csv('./在第一列添加幸存预测情况.csv')









