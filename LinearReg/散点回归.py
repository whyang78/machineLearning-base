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

def plotDataSet(xData,yData):
    xMat=np.array(xData)
    yMat=np.array(yData)

    x=xMat[:,1].T
    y=yMat
    plt.scatter(x,y,c='blue',alpha=0.5)
    plt.show()

#普通回归
def standRegres(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print('该矩阵为非奇异矩阵，不能求逆')
        return
    w=xTx.I*xMat.T*yMat
    return w

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m=xMat.shape[0]
    weights=np.mat(np.eye((m)))
    for i in range(m):
        temp=xMat[i,:]-testPoint
        weights[i,i]=np.exp(temp*temp.T/(-2*k**2))
    xTx=xMat.T*weights*xMat
    if np.linalg.det(xTx)==0.0:
        return
    w=xTx.I*xMat.T*weights*yMat
    return testPoint*w

def lwlrTest(testArr, xArr, yArr, k=1.0):
    testMat=np.mat(testArr)
    m=testMat.shape[0]
    y_pred=np.zeros(m)
    for i in range(m):
        y_pred[i]=lwlr(testMat[i],xArr,yArr,k)
    return y_pred

def plotRegression(xArr,yArr,w):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T

    x=xMat.copy()
    x.sort(0)
    y_pred=x*w
    plt.plot(x[:,1],y_pred,color='r')

    plt.scatter(xMat[:,1].A,yMat.A,c='blue',alpha=0.5)
    plt.show()

def plotlwlrRegression():
    xArr, yArr = loadDataSet('ex0.txt')                                    #加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)                            #根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)                            #根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)                            #根据局部加权线性回归计算yHat
    xMat = np.mat(xArr)                                                    #创建xMat矩阵
    yMat = np.mat(yArr)                                                    #创建yMat矩阵
    srtInd = xMat[:, 1].argsort(0)                                        #排序，返回索引值
    xSort = xMat[srtInd][:,0,:]
    fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'red')                        #绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'red')                        #绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'red')                        #绘制回归曲线
    axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)                #绘制样本点
    axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)                #绘制样本点
    axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)                #绘制样本点
    axs0_title_text=axs[0].set_title('k=1.0')
    axs1_title_text=axs[1].set_title('k=0.01')
    axs2_title_text=axs[2].set_title('k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    x,y=loadDataSet('./ex0.txt')
    # plotDataSet(x,y)
    # w=standRegres(x,y)
    # plotRegression(x,y,w)
    #
    # x=np.mat(x)
    # y=np.mat(y)
    # y_pred=x*w
    # print('预测值与真实值之间的相关系数：',np.corrcoef(y_pred.T,y))

    plotlwlrRegression()










