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

def linearSolve(dataSet):
    m,n=dataSet.shape
    one=np.mat(np.ones((m,1)))
    x=np.mat(np.concatenate((one,dataSet[:,:-1]),axis=1))
    y=dataSet[:,-1]

    xTx=x.T*x
    if np.linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the')
    w=xTx.I*x.T*y
    return w,x,y

def modelLeaf(dataSet):
    w,x,y=linearSolve(dataSet)
    return w

def modelErr(dataSet):
    w, x, y = linearSolve(dataSet)
    y_pred=x*w
    return np.sum(np.power((y_pred-y),2))

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


if __name__ == '__main__':
    # plotDataSet('./exp2.txt')

    dataset=np.mat(loadDataSet('./exp2.txt'))
    tree=createTree(dataset,modelLeaf,modelErr,(1,10))
    print(tree)

    #画出拟合图
    datamat=np.array(loadDataSet('./exp2.txt'))
    plt.scatter(datamat[:,0],datamat[:,1],c='blue',alpha=0.5)

    left=tree['leftTree']
    right=tree['rightTree']
    edge_val=float(tree['bestVal'])
    min_val=float(np.min(dataset[:,0],axis=0))
    max_val=float(np.max(dataset[:,0],axis=0))

    rw0=float(right.A[0])
    rw1=float(right.A[1])
    lw0=float(left.A[0])
    lw1=float(left.A[1])
    plt.plot([min_val,edge_val],[rw0+rw1*min_val,rw0+rw1*edge_val],color='r')
    plt.plot([edge_val,max_val],[lw0+lw1*edge_val,lw0+lw1*max_val],color='r')
    plt.show()


