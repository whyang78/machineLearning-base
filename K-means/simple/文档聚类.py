'''
1、加载数据集
'''
from sklearn.datasets import load_files
docs=load_files('./data')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_df=0.4, #一个单词在40%的文档中出现过 去除  最大
                           min_df=2, #一个单词只在不超过2个文档里出现 去除  最小
                           max_features=20000, #最大特征数，方便按权重选词
                           encoding='latin-1')
x=vectorizer.fit_transform((d for d in docs.data))

'''
2、文本聚类分析
'''
from sklearn.cluster import KMeans

clf=KMeans(n_clusters=4,
           n_init=3,
           max_iter=100, #最多进行100次k均值运算
           tol=0.01, #当中心点的移动小于0.01时，认为算法已经收敛，停止运算
           verbose=1 #显示运算过程
)
clf.fit(x)
# y=docs.target
# y_pred=clf.labels_
# centers=clf.cluster_centers_

from sklearn import metrics
labels = docs.target
#聚类效果检验
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, clf.labels_)) #齐次性
print("Completeness: %0.3f" % metrics.completeness_score(labels, clf.labels_)) #完整性
print("V-measure: %0.3f" % metrics.v_measure_score(labels, clf.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, clf.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x, clf.labels_, sample_size=1000)) #轮廓系数

'''
3、查看高频词
'''
order_centroids = clf.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(4):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()