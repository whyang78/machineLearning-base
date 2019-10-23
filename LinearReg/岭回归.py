import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    xMat=[]
    yMat=[]
    with open(fileName,'r+') as f:
        content=f.readlines()
        for line in content:
            line_arr=line.strip().split('\t')
            xMat.append(list(map(float,line_arr[:-1])))
            yMat.append(float(line_arr[-1]))
    return xMat,yMat

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx=xMat.T*xMat
    temp=xTx+np.eye(xMat.shape[1])*lam
    if np.linalg.det(temp)==0:
        return
    w=temp.I*xMat.T*yMat
    return w

#对比不同lamda条件下，回归系数的变化
#lamda越大，惩罚力度越大，回归系数趋于零；反之，非零系数特征数较多
def ridgeTest(xArr, yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T

    #数据标准化
    y_mean=np.mean(yMat,axis=0)
    yMat=yMat-y_mean
    x_mean=np.mean(xMat,axis=0)
    x_var=np.var(xMat,axis=0)
    xMat=(xMat-x_mean)/x_var

    numItem=30
    wMat=np.zeros((numItem,xMat.shape[1]))
    for i in range(30):
        w=ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=w.T

    return wMat

def plotwMat():
    x,y=loadDataSet('./abalone.txt')
    wMat=ridgeTest(x,y)
    plt.plot(wMat)
    plt.xlabel('lamda')
    plt.ylabel('w')
    plt.show()

if __name__ == '__main__':
    plotwMat()