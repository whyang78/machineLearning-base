import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    data=[]
    with open(fileName,'r+') as f:
        content=f.readlines()
        for line in content:
            line_arr=line.strip().split('\t')
            data.append(list([float(line_arr[4]),float(line_arr[3])]))
    return data

#欧氏距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power((vecA-vecB),2)))

#球面距离余弦公式计算距离
def distSLC(vecA, vecB):
    # 经度和维度用角度作为单位,但是sin()和cos()以弧度为输入.
    # 可以将江都除以180度然后再诚意圆周率pi转换为弧度
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * \
        np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0

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

def biKmeans(dataMat, k, distMeas=distEclud):
    m,n=dataMat.shape
    clusterAssment=np.mat(np.zeros((m,2)))
    center0=np.mean(dataMat,axis=0).tolist()[0]
    center_list=[center0]
    for i in range(m):
        clusterAssment[i,1]=distMeas(np.mat(center0),dataMat[i,:])**2

    while len(center_list)<k:
        minSSE=np.inf
        for i in range(len(center_list)):
            datamat_i=dataMat[np.nonzero(clusterAssment[:,0].A==i)[0]]
            split_centers, split_clusterAssment=kMeans(datamat_i,2,distMeas)
            split_SSE=np.sum(split_clusterAssment[:,1])
            no_split_SSE=np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0]])
            total_SSE=split_SSE+no_split_SSE
            if total_SSE<minSSE:
                minSSE=total_SSE
                bestsplitcenter=i
                bestclusterAssment=split_clusterAssment.copy()
                bestcenters=split_centers.copy()
        bestclusterAssment[np.nonzero(bestclusterAssment[:,0].A==1)[0],0]=len(center_list)
        bestclusterAssment[np.nonzero(bestclusterAssment[:,0].A==0)[0],0]=bestsplitcenter

        center_list[bestsplitcenter]=bestcenters[0,:].tolist()[0]
        center_list.append(bestcenters[1,:].tolist()[0])

        clusterAssment[np.nonzero(clusterAssment[:,0].A==bestsplitcenter)[0]]=bestclusterAssment

    return np.mat(center_list),clusterAssment

if __name__ == '__main__':
    datamat=np.mat(loadDataSet('./places.txt'))
    k=5
    centers,clusterAssment=biKmeans(datamat,k,distMeas=distSLC)

    fig=plt.figure()
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    rect=[0.1, 0.1, 0.8, 0.8]
    ax0 = fig.add_axes(rect, label='ax0',xticks=[],yticks=[])
    img=plt.imread('./Portland.png')
    ax0.imshow(img)

    # 叠加图层时frameon必须设置成False，不然会覆盖下面的图层
    ax1=fig.add_axes(rect, label='ax0',frameon=False)
    for i in range(k):
        data_i=datamat.A[np.nonzero(clusterAssment[:,0].A==i)[0]]
        ax1.scatter(data_i[:,0],data_i[:,1],s=100,marker=scatterMarkers[i])
    ax1.scatter(centers.A[:,0],centers.A[:,1],s=200,c='r',marker='+')
    plt.show()




