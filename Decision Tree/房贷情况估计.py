import numpy as np
from math import log
from collections import Counter
import pickle

#制作数据集,类型list
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels


def calcShannonEnt(dataSet):
    labels=[i[-1] for i in dataSet]
    total=len(labels)
    labels_dict=Counter(labels)

    entropy=0.0
    for _,value in labels_dict.items():
        prob=float(value)/float(total)
        entropy-=prob*log(prob,2)

    return entropy

def splitDataSet(dataSet, feat_index, value):
    #此处采用numpy的方法 注意元素数据类型
    dataSet_mat=np.mat(dataSet)
    feat_dataset=dataSet_mat[np.nonzero(dataSet_mat[:,feat_index]==str(value)),:][0]
    feat_dataset=np.delete(feat_dataset,feat_index,axis=1).tolist()
    feat_dataset_1=[]
    for i in feat_dataset:
        a=list(map(int,i[:-1]))
        a.append(i[-1])
        feat_dataset_1.append(a)
    return feat_dataset_1

def chooseBestFeatureToSplit(dataSet):
    num_feature=len(dataSet[0])-1
    base_entropy=calcShannonEnt(dataSet)
    bestfeat=-1
    best_entropy_gain=0.0
    for feat_index in range(num_feature):
        feat_value=set([i[feat_index] for i in dataSet])
        feat_entropy=0.0
        for value in feat_value:
            feat_dataset=splitDataSet(dataSet,feat_index,value)
            prob=float(len(feat_dataset))/float(len(dataSet))
            feat_entropy+=prob*calcShannonEnt(feat_dataset)
        entropy_gain=base_entropy-feat_entropy
        if entropy_gain>best_entropy_gain:
            best_entropy_gain=entropy_gain
            bestfeat=feat_index
    return bestfeat

def majorityCnt(classList):
    majorClass=Counter(classList).most_common(1)[0][0]
    return majorClass

def createTree(dataSet, labels, featLabels):
    classList=[i[-1] for i in dataSet]
    #停止条件：1、所有类标签完全一致 2、所有特征使用完毕，选择出现次数最多的类别
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(labels)==0 or len(dataSet[0])==1:
        return majorityCnt(classList)

    best_feat_index=chooseBestFeatureToSplit(dataSet)
    best_feat=labels[best_feat_index]
    featLabels.append(best_feat)

    myTree={best_feat:{}}
    del(labels[best_feat_index]) #删除已用特征标签
    feat_value=set([i[best_feat_index] for i in dataSet])
    for value in feat_value:
        myTree[best_feat][value]=createTree(splitDataSet(dataSet,best_feat_index,value),
                                            labels,featLabels)
    return myTree

def classify(myTree,testVec,labels):
    root=next(iter(myTree))
    second_dict=myTree[root]
    feat_index=labels.index(root)

    for key in second_dict.keys():
        if testVec[feat_index]==key:
            if type(second_dict[key]).__name__=='dict':
                return classify(second_dict[key],testVec,labels)
            else:
                return second_dict[key]

def save_myTree(file_name,myTree):
    with open(file_name,'wb') as f:
        pickle.dump(myTree,f)

def load_myTree(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    dataSet, labels=createDataSet()
    origin_labels=labels.copy()#因为在生成树过程中会不断删减特征标签，故初始时另存一个

    featLabels=[] #生成树过程中有用的特征标签，按序出现
    myTree=createTree(dataSet,labels,featLabels)

    save_myTree('./myTree.txt',myTree)
    print(load_myTree('./myTree.txt'))

    test_vec=[1,0,0,2] #对应于origin_labels
    result=classify(myTree,test_vec,origin_labels)
    print(result)
