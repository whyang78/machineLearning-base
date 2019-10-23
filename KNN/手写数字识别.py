import numpy as np
from collections import Counter
import operator
import os
from sklearn.neighbors import KNeighborsClassifier as knn
train_digits='./digits/trainingDigits'
test_digits='./digits/testDigits'

def classify0(inx,dataset,labels,k):
    dist=np.sum((inx-dataset)**2,axis=1)**0.5

    k_labels=[labels[i] for i in dist.argsort()[:k]]
    label_dict={}
    for i in k_labels:
        label_dict[i]=label_dict.get(i,0)+1
    label_list=sorted(label_dict.items(),key=operator.itemgetter(1),reverse=True)

    return label_list[0][0]

# def classify0(inx,dataset,labels,k):
#     #计算距离
#     dist=np.sum((inx-dataset)**2,axis=1)**0.5
#
#     #找到距离最近的前k个label
#     k_labels=[labels[i] for i in dist.argsort()[:k]]
#
#     #归类 counter返回类型list,内部元素为tuple
#     label=Counter(k_labels).most_common(1)[0][0]
#     return label

def img2vector(filename):
    with open(filename,'r+') as f:
        returnVec=list(map(int,f.read().replace('\n','')))
    return returnVec

def handwritingClassTest():
    #获取训练特征
    train_mat=[]
    train_labels=[]
    for sub in os.listdir(train_digits):
        label=sub.split('_')[0]
        train_labels.append(label)

        vector=img2vector(os.path.join(train_digits,sub))
        train_mat.append(vector)
    train_mat=np.array(train_mat)

    #进行测试
    error=0.0
    total=0.0
    for sub in os.listdir(test_digits):
        total+=1
        label=sub.split('_')[0]
        vector=img2vector(os.path.join(test_digits,sub))

        pred_label=classify0(vector,train_mat,train_labels,3)
        if pred_label!=label:
            error+=1

    print('错误率为{}/{}'.format(error,total))

def sklearn_handwritingTest():
    #获取训练特征
    train_mat=[]
    train_labels=[]
    for sub in os.listdir(train_digits):
        label=sub.split('_')[0]
        train_labels.append(label)

        vector=img2vector(os.path.join(train_digits,sub))
        train_mat.append(vector)
    train_mat=np.array(train_mat)

    neigh=knn(n_neighbors=3)
    neigh.fit(train_mat,train_labels)

    #进行测试
    error=0.0
    total=0.0
    for sub in os.listdir(test_digits):
        total+=1
        label=sub.split('_')[0]
        vector=img2vector(os.path.join(test_digits,sub))

        vector=np.array(vector).reshape(1,-1)# 1d->2d
        pred_label=neigh.predict(vector)
        if pred_label!=label:
            error+=1

    print('错误率为{}/{}'.format(error, total))

if __name__ == '__main__':
    handwritingClassTest()
    sklearn_handwritingTest()



