import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负

def file2matrix(filePath):
    with open(filePath,'r+') as f:
        content=f.readlines()
        retMat=np.zeros([len(content),3])
        labels=[]

        index=0
        for line in content:
            line=line.strip().split('\t')
            #使用map对可迭代数据进行类型转换
            retMat[index,:]=list(map(float,line[:3]))
            labels.append(line[3])
            index+=1

        return retMat,labels

#将txt存储的文件转换为csv存储
def file2csv(filePath,name):
    with open(filePath,'r+') as f:
        content=f.readlines()

        li=[]
        for line in content:
            line=list(line.strip().split('\t'))
            li.append(line)

        df=pd.DataFrame(li)
        df.to_csv(name,header=0,index=0)#无索引，无列名称

def autonorm(dataset):
    maxVal=np.max(dataset,axis=0)
    minVal=np.min(dataset,axis=0)
    val_range=maxVal-minVal

    normDataset=((dataset-minVal)/val_range).astype(np.float)
    return normDataset,val_range,minVal

def classify0(inx,dataset,labels,k):
    dist=np.sum((inx-dataset)**2,axis=1)**0.5
    k_labels=[labels[i] for i in dist.argsort()[:k]]
    label=Counter(k_labels).most_common(1)[0][0]
    return label

def datingTest():
    test_radio=0.1
    retMat, labels = file2matrix('./datingTestSet2.txt')
    normDataset, val_range, minVal = autonorm(retMat)

    n=retMat.shape[0]
    testNum=int(n*test_radio)
    error=0.0
    k=3
    for i in range(testNum):
        predLabel=classify0(normDataset[i,:],normDataset[testNum:,:],labels[testNum:],k)
        if predLabel!=labels[i]:
            error+=1.0

    print('错误率为%.2f%%'%(error/float(testNum)*100))

def showdatas(datingDataMat, datingLabels):
	#将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
	#当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False)

	numberOfLabels = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if int(i) == 1:
			LabelsColors.append('black')
		if int(i) == 2:
			LabelsColors.append('orange')
		if int(i) == 3:
			LabelsColors.append('red')
	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比')
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
	axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')
	plt.setp(axs1_title_text, size=9, weight='bold', color='red')
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

	#画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
	axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
	plt.setp(axs2_title_text, size=9, weight='bold', color='red')
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
	#设置图例
	didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',
	                  markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',
	                  markersize=6, label='largeDoses')
	#添加图例
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
	#显示图片
	plt.show()

if __name__ == '__main__':
    # file2csv('./datingTestSet2.txt','./datingTestSet2.csv')
    # datingTest()
    retMat, labels = file2matrix('./datingTestSet2.txt')
    showdatas(retMat,labels)



