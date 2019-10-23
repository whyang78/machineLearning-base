import numpy as np
import random
from sklearn.linear_model import LogisticRegression
'''
1、使用自己写的函数进行预测
'''
def sigmoid(inX):
    return 1.0/(1.0+np.exp(-inX))

#梯度上升法
def gradAscent(dataMatIn, classLabels):
    datamat=np.mat(dataMatIn)
    labelmat=np.mat(classLabels).transpose()
    m,n=datamat.shape

    weights=np.ones((n,1))
    alpha=0.01
    maxCycles=500
    for i in range(maxCycles):
        h=sigmoid(datamat*weights)
        weights=weights+alpha*datamat.transpose()*(labelmat-h)
    return weights

#随机梯度下降法，改进之处：1、alpha变化 2、每次随机选择一个样本更新梯度
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    datamat=np.mat(dataMatrix)
    m,n=datamat.shape

    weights=np.ones((n,1))
    for i in range(numIter):
        index=list(range(m))
        for j in range(m):
            randIndex=int(random.uniform(0,len(index)))
            alpha=0.01+4/(1.0+i+j)
            h=sigmoid(datamat[randIndex]*weights)
            weights=weights+alpha*datamat[randIndex].transpose()*(classLabels[randIndex]-h)
            del index[randIndex]
    return weights

def colicTest(Data,Label,weights):
    datamat = np.mat(Data)
    labelmat=np.mat(Label).transpose()

    pred=datamat*weights
    pred=(pred>0.5).astype(np.int)
    correct=np.equal(labelmat,pred).sum()
    print('正确比例：{}/{}'.format(correct,datamat.shape[0]))

def load_dataset(file_path):
    datalist=[]
    classlist=[]
    with open(file_path,'r+') as f:
        fr=f.readlines()
        for line in fr:
            linearr=line.strip().split('\t')
            datalist.append(list(map(float,linearr[:-1])))
            classlist.append(float(linearr[-1]))

    return datalist,classlist

if __name__ == '__main__':
    train_data,train_label=load_dataset('./horseColicTraining.txt')
    test_data,test_label=load_dataset('./horseColicTest.txt')

    #自己写的函数的预测
    weights_1=gradAscent(train_data,train_label)
    weights_2=stocGradAscent1(train_data,train_label)
    colicTest(test_data,test_label,weights_1)
    colicTest(test_data, test_label, weights_2)
    #这里可以发现FG性能优于SAG，因为数据集比较小
    #当数据集较小时，我们使用梯度上升算法
    #当数据集较大时，我们使用改进的随机梯度上升算法

    #在Sklearn中，我们就可以根据数据情况选择优化算法，
    #比如数据较小的时候，我们使用liblinear，数据较大时，我们使用sag和saga。
    clf=LogisticRegression(solver='sag',max_iter=2000)
    #此处次用liblinear就可以，若是采用sag,需要加大max_iter，否则无法收敛
    clf.fit(train_data,train_label)
    pred=clf.predict(test_data)
    correct=np.equal(pred,np.array(test_label)).sum()
    print('正确比例：{}/{}'.format(correct,len(test_label)))
    print(clf.score(test_data,test_label))
