import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    data=[[1.,  2.1],
	    [ 1.5,  1.6],
	    [ 1.3,  1. ],
	    [ 1. ,  1. ],
	    [ 2. ,  1. ]]
    labels=[1.0, 1.0, -1.0, -1.0, 1.0]
    return data,labels

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def showDataSet(data, label):
    dataMat=np.array(data)
    labelMat=np.array(label)
    data_plus=dataMat[labelMat==1.0,:]
    data_minus=dataMat[labelMat==-1.0,:]

    plt.scatter(data_plus[:,0],data_plus[:,1],c='r')
    plt.scatter(data_minus[:,0],data_minus[:,1],c='g')
    plt.show()
    return dataMat, labelMat, data_plus, data_minus

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retLabels=np.ones((dataMatrix.shape[0],1))
    if threshIneq=='lt':
        retLabels[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retLabels[dataMatrix[:,dimen]>threshVal]=-1.0
    return retLabels

def buildStump(dataArr, classLabels, D):
    dataMat=np.mat(dataArr)
    labelMat=np.mat(classLabels).T
    m,n=dataMat.shape

    bestClassLabels=np.mat(np.zeros((m,1)))
    bestStump={}
    min_error=float('inf')
    numsteps=10.0
    for i in range(n): #遍历所有特征
        rangeMin=min(dataMat[:,i])
        rangeMax=max(dataMat[:,i])
        step=(rangeMax-rangeMin)/numsteps
        for j in range(-1,int(numsteps)+1): #遍历所有特征取值，划分多个阈值
            threshVal=rangeMin+step*float(j)
            for threshIneq in ['lt','gt']: #遍历不等式
                predLabel=stumpClassify(dataMat,i,threshVal,threshIneq)
                error_arr=np.mat(np.zeros((m,1)))
                error_arr[predLabel!=labelMat]=1
                weight_error=D.T*error_arr
                if weight_error<min_error:
                    min_error=weight_error
                    bestClassLabels=predLabel.copy()
                    bestStump['dimen']=i
                    bestStump['threshVal']=threshVal
                    bestStump['threshIneq']=threshIneq
    return min_error,bestStump,bestClassLabels

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    dataMat=np.mat(dataArr)
    labelMat=np.mat(classLabels).T
    weakclassifier=[]
    finalClasslabel=np.mat(np.zeros((dataMat.shape[0],1)))
    D=np.mat(np.ones((dataMat.shape[0],1))/dataMat.shape[0])
    for iter in range(numIt):
        min_error, bestStump, bestClassLabels=buildStump(dataArr,classLabels,D)
        alpha=float(0.5*np.log((1-min_error)/max(min_error,1e-16)))
        bestStump['alpha']=alpha
        weakclassifier.append(bestStump)

        exp=np.multiply(-1*alpha*labelMat,bestClassLabels)
        D=np.multiply(D,np.exp(exp))
        D=D/sum(D)

        finalClasslabel+=alpha*bestClassLabels
        finalError=np.multiply(np.sign(finalClasslabel)!=labelMat,np.ones((dataMat.shape[0],1)))
        errorRate=sum(finalError)/finalError.shape[0]
        if errorRate==0:
            print('共有{}个弱分类器'.format(iter + 1))
            break
    return weakclassifier,finalClasslabel

def plotROC(predStrengths, classLabels):
        cur = (0.0, 0.0)  # 绘制光标的位置 从右下角开始绘 正例为1故降序 正例为-1则升序 绘出的图像对称
        ySum = 0.0  # 用于计算AUC
        numPosClas = np.sum(np.array(classLabels) == 1.0)  # 统计正类的数量
        yStep = 1 / float(numPosClas)  # y轴步长
        xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长

        sortedIndicies = (-1*predStrengths).argsort()  # 预测强度排序 argsort默认升序 原数据取反实现降序

        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        for index in sortedIndicies.tolist()[0]:
            if classLabels[index] == 1.0:
                delX = 0
                delY = yStep
            else:
                delX = xStep
                delY = 0
                ySum += cur[1]  # 高度累加
            ax.plot([cur[0], cur[0] + delX], [cur[1], cur[1] + delY], c='b')  # 绘制ROC
            cur = (cur[0] + delX, cur[1] + delY)  # 更新绘制光标的位置
        ax.plot([0, 1], [0, 1], 'b--')
        plt.title('ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        ax.axis([0, 1, 0, 1])
        print('AUC面积为:', ySum * xStep)  # 计算AUC
        plt.show()

if __name__ == '__main__':

    traindata,trainlabel=loadDataSet('./horseColicTraining2.txt')
    weakclassifier, finalClasslabel = adaBoostTrainDS(traindata, trainlabel,10)
    plotROC(finalClasslabel.T,trainlabel)








