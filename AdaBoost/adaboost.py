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
    return weakclassifier,np.sign(finalClasslabel)

def adaClassify(datToClass, classifierArr):
    retClass=np.mat(np.zeros((datToClass.shape[0],1)))
    for i in range(len(classifierArr)):
        alpha=classifierArr[i]['alpha']
        h=stumpClassify(datToClass,classifierArr[i]['dimen'],
                      classifierArr[i]['threshVal'],classifierArr[i]['threshIneq'])
        retClass+=alpha*h
        # print('第{}个弱分类器输出结果：{}'.format(i+1,retClass))
    return np.sign(retClass)

if __name__ == '__main__':
    # 散点测试集测试
    # data,label=loadSimpData()
    # weakclassifier,finalClasslabel=adaBoostTrainDS(data,label)
    # print(adaClassify(np.mat([[0,0],[5,5]]), weakclassifier))

    #病马疝气死亡率测试
    traindata,trainlabel=loadDataSet('./horseColicTraining2.txt')
    weakclassifier, finalClasslabel = adaBoostTrainDS(traindata, trainlabel)

    pred_train=adaClassify(np.mat(traindata),weakclassifier)
    error=(pred_train!=np.mat(trainlabel).T).sum()
    print('训练集错误率：{:.5f}'.format(float(error)/float(len(trainlabel))))

    testdata,testlabel=loadDataSet('./horseColicTest2.txt')
    pred_test=adaClassify(np.mat(testdata),weakclassifier)
    error=(pred_test!=np.mat(testlabel).T).sum()
    print('测试集错误率：{:.5f}'.format(float(error)/float(len(testlabel))))




