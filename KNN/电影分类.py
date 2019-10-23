import numpy as np
from collections import Counter

def createDataset():
    group=np.array([[3,104],
                    [2,100],
                    [1,81],
                    [101,10],
                    [99,5],
                    [98,2]])
    labels=['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    return group,labels

def classify0(inx,dataset,labels,k):
    #计算距离
    dist=np.sum((inx-dataset)**2,axis=1)**0.5

    #找到距离最近的前k个label
    k_labels=[labels[i] for i in dist.argsort()[:k]]

    #归类 counter返回类型list,内部元素为tuple
    label=Counter(k_labels).most_common(1)[0][0]
    return label

if __name__ == '__main__':
    group,labels=createDataset()

    k=3
    test=[18,90]
    label=classify0(test,group,labels,k)
    print(label)
