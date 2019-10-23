import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet(fileName):
    trainmat=[]
    labelmat=[]
    with open(fileName,'r+') as f:
        content=f.readlines()
        for line in content:
            line_arr=line.strip().split('\t')
            trainmat.append([float(line_arr[0]),float(line_arr[1])])
            labelmat.append(float(line_arr[2]))
    return trainmat,labelmat

def showDataSet(dataMat, labelMat):
    datamat,labelmat=np.array(dataMat),np.array(labelMat)
    datamat_plus=datamat[labelmat[:]==1,:]
    datamat_minus=datamat[labelmat[:]==-1,:]
    plt.scatter(datamat_plus[:,0],datamat_plus[:,1],c='r')
    plt.scatter(datamat_minus[:,0],datamat_minus[:,1],c='g')
    plt.show()

def selectJrand(i, m):
    j=i
    while j==i:
        j=random.uniform(0,m)
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    elif aj<L:
        aj=L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat=np.mat(dataMatIn) #100,2
    classmat=np.mat(classLabels).T #100,1
    m,n=dataMat.shape
    alphas=np.mat(np.zeros((m,1)))
    b=0
    iter=0
    while iter<maxIter:
        alphapairsChange=0
        for i in range(m):
            #计算误差
            i=int(i)
            Fxi=float(np.multiply(alphas,classmat).T*dataMat*dataMat[i,:].T)+b
            Ei=Fxi-float(classmat[i])
            #判断是否不符合KKT条件
            if((alphas[i]<C)and(labelmat[i]*Ei<-toler))or((alphas[i]>0)and(labelmat[i]*Ei>toler)):
                #随机选择j
                j=selectJrand(i,m)
                j=int(j)
                #计算j误差
                Fxj = float(np.multiply(alphas, classmat).T * dataMat * dataMat[j, :].T)+b
                Ej = Fxj - float(classmat[j])

                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                #计算上下界
                if classmat[i]!=classmat[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                if classmat[i]==classmat[j]:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print('L==H')
                    continue
                #计算eta 学习率 特殊情况eta=0，在此不作处理
                eta=dataMat[i,:]*dataMat[i,:].T+dataMat[j,:]*dataMat[j,:].T-2*dataMat[i,:]*dataMat[j,:].T
                if eta<=0:
                    print('eta<=0')
                    continue
                #更新aj
                alphas[j]+=classmat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print("alpha_j变化太小")
                    continue
                #更新ai
                alphas[i]+=classmat[i]*classmat[j]*(alphaJold-alphas[j])
                #更新b1,b2
                b1=b-Ei-classmat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T-classmat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[i,:].T
                b2=b-Ei-classmat[j]*(alphas[i]-alphaIold)*dataMat[j,:]*dataMat[i,:].T-classmat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T
                if alphas[i]>0 and alphas[i]<C:
                    b=b1
                elif alphas[j]>0 and alphas[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2
                alphapairsChange+=1
        print('迭代序号：{}，优化次数：{}'.format(iter,alphapairsChange))
        if alphapairsChange==0:
            iter+=1 #若没有参数更新，继续计数
        else:
            iter=0 #若存在参数更新，则重新计数
        #结束条件：在最大迭代次数中，不再更新任何参数
    return b,alphas

def get_w(dataMat, labelMat, alphas):
    data=np.mat(dataMat)
    label=np.mat(labelMat).T
    w=np.multiply(label,alphas).T*data
    return w

def showClassifer(dataMat,labelMat, w, b, alphas):
    datamat,labelmat=np.array(dataMat),np.array(labelMat)
    datamat_plus=datamat[labelmat[:]==1,:]
    datamat_minus=datamat[labelmat[:]==-1,:]
    plt.scatter(datamat_plus[:,0],datamat_plus[:,1],c='r')
    plt.scatter(datamat_minus[:,0],datamat_minus[:,1],c='g')

    w=np.array(w)
    w1,w2=w[:,0][0],w[:,1][0]
    x1=np.max(datamat[:,0],axis=0)
    x2=np.min(datamat[:,0],axis=0)
    b=np.array(b)
    b=b[:,0][0]
    y1=(-b-w1*x1)/w2
    y2=(-b-w1*x2)/w2
    plt.plot([x1,x2],[y1,y2])
    for i in range(len(alphas)):
        if alphas[i]>0:
            x,y=datamat[i]
            plt.scatter([x],[y],s=150,linewidths=1.5,edgecolors='blue')
    plt.show()


if __name__ == '__main__':
    datamat,labelmat=loadDataSet('./testSet.txt')
    b, alphas = smoSimple(datamat, labelmat, 0.6, 0.001, 40)
    w = get_w(datamat, labelmat, alphas)
    showClassifer(datamat,labelmat,w,b,alphas)