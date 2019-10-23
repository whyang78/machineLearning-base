import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负

def loadDataSet():
    datamat=[]
    labelmat=[]
    with open('./testSet.txt','r+') as f:
        fr=f.readlines()
        for line in fr:
            line_arr=line.strip().split()
            datamat.append([1.0,float(line_arr[0]),float(line_arr[1])])
            labelmat.append(int(line_arr[2]))
    return datamat,labelmat

def sigmoid(inX):
    return 1.0/(1.0+np.exp(-inX))

#梯度上升法
def gradAscent(dataMatIn, classLabels):
    datamat=np.mat(dataMatIn) #100,3
    labelmat=np.mat(classLabels).transpose() #100,1
    m,n=datamat.shape

    weights=np.ones((n,1))
    alpha=0.01
    maxCycles=500
    weights_arr=np.array([])
    for i in range(maxCycles):
        h=sigmoid(datamat*weights)
        weights=weights+alpha*datamat.transpose()*(labelmat-h)
        weights_arr=np.append(weights_arr,weights)
    weights_arr=weights_arr.reshape(maxCycles,-1)
    return weights,weights_arr

#随机梯度下降法，改进之处：1、alpha变化 2、每次随机选择一个样本更新梯度
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    datamat=np.mat(dataMatrix)
    m,n=datamat.shape

    weights=np.ones((n,1))
    weights_arr=np.array([])
    for i in range(numIter):
        index=list(range(m))
        for j in range(m):
            randIndex=int(random.uniform(0,len(index)))
            alpha=0.01+4/(1.0+i+j)
            h=sigmoid(datamat[randIndex]*weights)
            weights=weights+alpha*datamat[randIndex].transpose()*(classLabels[randIndex]-h)
            weights_arr=np.append(weights_arr,weights)
            del index[randIndex]
    weights_arr=weights_arr.reshape(numIter*m,-1)
    return weights,weights_arr

def plotBestFit(dataMatrix, classLabels,weights):
    datamat=np.mat(dataMatrix)
    labelmat=np.array(classLabels)
    x=datamat[:,1]
    y=datamat[:,2]
    x0,x1=x[labelmat[:]==0,:],x[labelmat[:]==1,:]
    y0,y1=y[labelmat[:]==0,:],y[labelmat[:]==1,:]

    plt.scatter(list(x0),list(y0),c='g',s=20,marker='o')
    plt.scatter(list(x1),list(y1),c='r',s=20,marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def plotWeights(weights_array1,weights_array2):

	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
	fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
	x1 = np.arange(0, len(weights_array1), 1)
	#绘制w0与迭代次数的关系
	axs[0][0].plot(x1,weights_array1[:,0])
	axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系')
	axs0_ylabel_text = axs[0][0].set_ylabel(u'W0')
	plt.setp(axs0_title_text, size=20, weight='bold', color='black')
	plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
	#绘制w1与迭代次数的关系
	axs[1][0].plot(x1,weights_array1[:,1])
	axs1_ylabel_text = axs[1][0].set_ylabel(u'W1')
	plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
	#绘制w2与迭代次数的关系
	axs[2][0].plot(x1,weights_array1[:,2])
	axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数')
	axs2_ylabel_text = axs[2][0].set_ylabel(u'W1')
	plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
	plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


	x2 = np.arange(0, len(weights_array2), 1)
	#绘制w0与迭代次数的关系
	axs[0][1].plot(x2,weights_array2[:,0])
	axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系')
	axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
	plt.setp(axs0_title_text, size=20, weight='bold', color='black')
	plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
	#绘制w1与迭代次数的关系
	axs[1][1].plot(x2,weights_array2[:,1])
	axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
	plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
	#绘制w2与迭代次数的关系
	axs[2][1].plot(x2,weights_array2[:,2])
	axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数')
	axs2_ylabel_text = axs[2][1].set_ylabel(u'W1')
	plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
	plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

	plt.show()

def colicTest(Data,Label,weights):
    datamat=np.mat(Data) #100,3
    labelmat=np.mat(Label).transpose() #100,1

    pred=datamat*weights
    pred=(pred>0.5).astype(np.int)
    correct=np.equal(labelmat,pred).sum()
    print('正确比例：{}/{}'.format(correct,datamat.shape[0]))



if __name__ == '__main__':
    data, label=loadDataSet()
    weights_1,weights_arr_1=gradAscent(data,label)
    weights_2, weights_arr_2 = stocGradAscent1(data, label)
    plotWeights(weights_arr_2,weights_arr_1)
    colicTest(data,label,weights_1)
    colicTest(data,label,weights_2)

