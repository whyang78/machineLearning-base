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

def regularize(xMat, yMat):
    x_mean=np.mean(xMat,axis=0)
    x_var=np.var(xMat,axis=0)
    x=(xMat-x_mean)/x_var

    y_mean=np.mean(yMat,axis=0)
    y=yMat-y_mean
    return x,y

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xMat,yMat=regularize(xMat,yMat)

    wMat=np.zeros((numIt,xMat.shape[1]))
    w=np.zeros((xMat.shape[1],1))
    w_max=w.copy()
    for iter in range(numIt):
        minError=float('inf')
        for i in range(xMat.shape[1]):
            for j in [1,-1]:
                w_test=w.copy()
                w_test[i]+=j*eps
                y_pred=xMat*w_test
                test_error=rssError(yMat.A,y_pred.A)
                if test_error<minError:
                    minError=test_error
                    w_max=w_test
        w=w_max.copy()
        wMat[iter,:]=w.T
    return wMat

def plotstageWiseMat():
    x,y=loadDataSet('./abalone.txt')
    wMat=stageWise(x,y,0.005,1000)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(wMat)
    title=ax.set_title('LinearREG')
    xlabel=ax.set_xlabel('times')
    ylabel=ax.set_ylabel('w')
    plt.setp(title,size=15,color='red')
    plt.setp(xlabel,size=10,color='black')
    plt.setp(ylabel,size=10,color='black')
    plt.show()

if __name__ == '__main__':
    plotstageWiseMat()