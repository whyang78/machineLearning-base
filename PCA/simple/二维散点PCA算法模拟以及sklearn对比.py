'''
1、numpy模拟算法
'''
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[3, 2000],
              [2, 3000],
              [4, 5000],
              [5, 8000],
              [1, 2000]], dtype='float')

#数据归一化及缩放
mean=np.mean(data,axis=0)
norm=data-mean

scope=np.max(data,axis=0)-np.min(data,axis=0)
norm=norm/scope

#PCA算法 2D-->1D
U,S,V=np.linalg.svd(np.dot(norm.T,norm))
# u,s,v=np.linalg.svd(norm)
# print(U)
# print(v)
'''
[[-0.67710949 -0.73588229]
 [-0.73588229  0.67710949]]
 
[[-0.67710949 -0.73588229]
 [-0.73588229  0.67710949]]
'''
Ur=U[:,0].reshape(-1,1)
Z=np.dot(norm,Ur)

#数据还原
data_approx=np.dot(Z,Ur.T)
data_approx=data_approx*scope+mean

#绘制图像
data_min=np.min(data_approx,axis=0)
data_max=np.max(data_approx,axis=0)
plt.figure()
plt.scatter(data[:,0],data[:,1],s=20,c='blue',marker='o',alpha=0.2)
plt.scatter(data_approx[:,0],data_approx[:,1],s=20,c='red',marker='x')
plt.plot([data_min[0],data_max[0]],[data_min[1],data_max[1]],color='g')
plt.show()

'''
2、sklearn使用PCA
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def std_pca(**kwargs):
    Scaler=MinMaxScaler()
    pca=PCA(**kwargs)
    model=Pipeline([('Scaler',Scaler),
                    ('pca',pca)])
    return model

pca=std_pca(n_components=1)
Zs=pca.fit_transform(data)
data_approx_s=pca.inverse_transform(Zs)