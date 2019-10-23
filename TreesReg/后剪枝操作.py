import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    datamat=[]
    with open(fileName,'r+') as f:
        content=f.readlines()
        for line in content:
            line_arr=line.strip().split('\t')
            datamat.append(list(map(float,line_arr)))
    return datamat

def plotDataSet(filename):
    data=loadDataSet(filename)
    datamat=np.array(data)
    plt.scatter(datamat[:,0],datamat[:,1],c='blue',alpha=0.5)
    plt.show()

def binSplitDataSet(dataSet, feature, value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1])*dataSet.shape[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tols=ops[0]
    toln=ops[1]
    #若是数值一致，则返回叶节点
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)

    m,n=dataSet.shape
    base_error=errType(dataSet)
    bestFeat=0
    bestVal=0
    bestError=float('inf')
    for feat in range(n-1): #遍历所有特征
        for val in set(dataSet[:,feat].T.tolist()[0]):#遍历该特征所有值
            mat0,mat1=binSplitDataSet(dataSet,feat,val)
            #判断是否满足最小样本数限制
            if mat0.shape[0]<toln or mat1.shape[0]<toln:
                continue
            new_error=errType(mat0)+errType(mat1)
            if new_error<bestError:
                bestError=new_error
                bestFeat=feat
                bestVal=val
    #判断是否满足最小误差变化限制
    if (base_error-bestError)<tols:
        return None,leafType(dataSet)

    mat0,mat1=binSplitDataSet(dataSet,bestFeat,bestVal)
    #判断是否满足最小样本数限制
    if mat0.shape[0]<toln or mat1.shape[0]<toln:
        return None,leafType(dataSet)
    return bestFeat,bestVal

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    tree={}
    tree['bestFeat']=feat
    tree['bestVal']=val
    mat0,mat1=binSplitDataSet(dataSet,feat,val)
    #递归生成树
    tree['leftTree']=createTree(mat0,leafType,errType,ops)
    tree['rightTree']=createTree(mat1,leafType,errType,ops)
    return tree

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['leftTree']):
        tree['leftTree']=getMean(tree['leftTree'])
    if isTree(tree['rightTree']):
        tree['rightTree']=getMean(tree['rightTree'])
    return (tree['leftTree']+tree['rightTree'])/2.0

def prune(tree, testData):
    if testData.shape[0]==0:
        return getMean(tree)
    if isTree(tree['leftTree']) or isTree(tree['rightTree']):
        lset,rset=binSplitDataSet(testData,tree['bestFeat'],tree['bestVal'])
    #左子树剪枝
    if isTree(tree['leftTree']):
        tree['leftTree']=prune(tree['leftTree'],lset)
    #右子树剪枝
    if isTree(tree['rightTree']):
        tree['rightTree'] = prune(tree['rightTree'], rset)
    #找到最后一个分类节点
    if not isTree(tree['leftTree']) and not isTree(tree['rightTree']):
        lset, rset = binSplitDataSet(testData, tree['bestFeat'], tree['bestVal'])
        error_no_merge=np.sum(np.power((lset[:,-1]-tree['leftTree']),2))+\
                        np.sum(np.power((rset[:,-1]-tree['rightTree']),2))
        tree_mean=(tree['leftTree']+tree['rightTree'])/2.0 #塌陷处理
        error_merge=np.sum(np.power((testData[:,-1]-tree_mean),2))
        if error_merge<error_no_merge:
            return tree_mean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
   traindata=loadDataSet('./ex2.txt')
   tree=createTree(np.mat(traindata))
   print(tree)

   testdata=loadDataSet('./ex2test.txt')
   tree_cut=prune(tree,np.mat(testdata))
   print(tree_cut)

