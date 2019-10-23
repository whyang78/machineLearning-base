import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    data=[]
    with open(fileName,'r+') as f:
        content=f.readlines()
        for line in content:
            line_arr=line.strip().split('\t')
            data.append(list(map(float,line_arr)))
    return data

#欧氏距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power((vecA-vecB),2)))

def randCent(dataMat, k):
    m,n=dataMat.shape
    centers=np.mat(np.zeros((k,n)))
    for i in range(n):
        min_i=min(dataMat[:,i])
        range_i=float(max(dataMat[:,i])-min(dataMat[:,i]))
        centers[:,i]=np.mat(min_i+range_i*np.random.rand(k,1))
    return centers


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m,n=dataMat.shape
    centers=createCent(dataMat,k)
    #clusterAssment包含两个列:一列记录簇索引值,
    #第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m, 2)))
    changed=True
    while changed:
        changed=False
        for i in range(m):
            min_dist=np.inf
            min_index=-1
            for j in range(k):
                dist=distMeas(dataMat[i,:],centers[j,:])
                if dist<min_dist:
                    min_dist=dist
                    min_index=j
            #停止条件：所有点质心都不再发生变化
            if clusterAssment[i,0]!=min_index:
                changed=True
            clusterAssment[i,:]=min_index,min_dist**2
        #更新质心
        for i in range(k):
            data=dataMat[np.nonzero(clusterAssment[:,0].A==i)[0]]
            centers[i,:]=np.mean(data,axis=0)
    return centers,clusterAssment

def plot_dataset(datamat):
    data=np.array(datamat)
    plt.scatter(data[:,0],data[:,1],s=20,c='blue',alpha=0.5)
    plt.show()

def plot_cluster(datamat,centers,clusterAssment):
    k=len(centers)
    #绘制分类点
    color=['r','g','b','y']
    data=np.array(datamat)
    for i in range(k):
        data_i=data[clusterAssment.A[:,0]==i,:]
        plt.scatter(data_i[:, 0], data_i[:, 1], s=20,c=color[i], alpha=0.5)

    for i in range(k):
        center=np.array(centers)
        plt.scatter(center[i,0],center[i,1],s=100,c='black',marker='+')

    plt.show()

if __name__ == '__main__':
    datamat=np.mat(loadDataSet('./testSet.txt'))
    # plot_dataset(datamat)
    centers, clusterAssment=kMeans(datamat,k=4)
    plot_cluster(datamat,centers,clusterAssment)

