import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
'''
1、制作数据集
'''
x, y = make_blobs(n_samples=200,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)

clf=KMeans(n_clusters=3)
clf.fit_predict(x)
y_pred=clf.labels_ #聚类标签

from sklearn.metrics import v_measure_score,completeness_score,homogeneity_score
score1=v_measure_score(y,y_pred)
score2=completeness_score(y,y_pred)#完整性检验聚类效果
score3=homogeneity_score(y,y_pred)#齐次性检验聚类效果

'''
2、knn 采用不同的k
'''
def fit_plot_kmean_model(n_clusters, x):
    plt.xticks([])
    plt.yticks([])
    markers = ['o', '^', '*', 's']
    colors = ['r', 'b', 'y', 'k']

    clf=KMeans(n_clusters=n_clusters)
    clf.fit_predict(x)
    score=clf.score(x) #绝对值越大越不好
    plt.title('k={},score={:.2f}'.format(n_clusters,score))

    labels=clf.labels_
    centers=clf.cluster_centers_
    for i in range(n_clusters):
        cluster=x[labels==i]
        plt.scatter(cluster[:,0],cluster[:,1],s=30,c=colors[i],marker=markers[i])
    plt.scatter(centers[:,0],centers[:,1],s=200,c='white',marker='o',alpha=0.9)
    for i,c in enumerate(centers):
        plt.scatter(c[0],c[1],s=50,marker='$%d$'%i,c=colors[i])

n_cluster=[2,3,4]
plt.figure(figsize=(18,4))
for i,k in enumerate(n_cluster,1):
    plt.subplot(1,len(n_cluster),i)
    fit_plot_kmean_model(k,x)
plt.show()


