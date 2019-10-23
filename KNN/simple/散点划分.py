import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier

#生成具有一定中心点的散点数据集
centers =np.array([[-2, 2], [2, 2], [0, 4]])
x,y=make_blobs(n_samples=60,n_features=2,centers=centers,cluster_std=0.6,random_state=78)

#绘制图像
# plt.figure()
# plt.scatter(x[:,0],x[:,1],s=100,c=y,cmap='cool')
# plt.scatter(centers[:,0],centers[:,1],s=100,c='orange',marker='^')
# plt.show()

#算法归类，并绘制图像
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)

x_test=np.array([[0,2]])
y_pred=knn.predict(x_test)
#返回训练样本x中n_neighbors距离最近的点,返回数据类型array
neighbors=knn.kneighbors(x_test,return_distance=False)

plt.figure()
plt.scatter(x[:,0],x[:,1],s=100,c=y,cmap='cool')
plt.scatter(centers[:,0],centers[:,1],s=100,c='orange',marker='^')
plt.scatter(x_test[:,0],x_test[:,1],s=100,cmap='cool',marker='x')
for i in neighbors[0]:
    plt.plot( [x[i][0],x_test[0][0]] , [x[i][1],x_test[0][1]] ,'k--',linewidth=0.6)
plt.show()

