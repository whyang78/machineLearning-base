import numpy as np
import matplotlib.pyplot as plt

'''
1、获取数据集
'''
from sklearn.datasets import load_files

train_path='./dataset/train'
news_train=load_files(train_path)
print('文档总数：{},类别总数：{}'.format(len(news_train.data),len(news_train.target_names)))

'''
2、文档的数学表达 采用TF-IDF统计方法
'''
from sklearn.feature_extraction.text import TfidfVectorizer

TfVector=TfidfVectorizer(encoding='latin-1')
x_train=TfVector.fit_transform((i for i in news_train.data))
y_train=news_train.target
print('样本总数：%d，特征总数：%d'%x_train.shape)
print('样本{}的非零特征总数为{}'.format(news_train.filenames[0],x_train[0].getnnz()))

# file_path='./dataset/train/alt.atheism/0-53225'
# def check_fileDE(path):
#     import chardet
#     with open(path,'rb') as f:
#         data=f.read()
#         DEtype=chardet.detect(data)
#     return DEtype
#
# print(check_fileDE(file_path))

'''
3、模型训练 测试
'''
from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import GridSearchCV
#
# clf=GridSearchCV(MultinomialNB(),{'alpha':np.linspace(0,0.1,100)},
#                  cv=5,return_train_score=True)
# clf.fit(x_train,y_train)
# print('多项式模型最优参数：{},训练分数：{}'.format(clf.best_params_,clf.best_score_))
# 'alpha': 0.007070707070707071

#根据最优参数进行测试集测试
test_path='./dataset/test'
news_test=load_files(test_path)
x_test=TfVector.transform((i for i in news_test.data))
y_test=news_test.target

clf_test=MultinomialNB(alpha=0.007070707070707071)
clf_test.fit(x_train,y_train)
pred_test=clf_test.predict(x_test)
train_score=clf_test.score(x_train,y_train)
test_score=clf_test.score(x_test,y_test)
print('最优参数模型训练分数：{}，测试分数：{}'.format(train_score,test_score))

'''
4、模型评价 采用分类报告、混淆矩阵
'''
from sklearn.metrics import classification_report,confusion_matrix

clf_report=classification_report(y_test,pred_test,target_names=news_test.target_names)
print(clf_report)

cm=confusion_matrix(y_test,pred_test)
plt.figure()
plt.title('Confusion matrix of the classifier')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.matshow(cm, fignum=1, cmap='Reds')
plt.colorbar()
plt.show()


