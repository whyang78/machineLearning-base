'''
1、dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
2、 python3中用items()替换python2中的iteritems()
	key=operator.itemgetter(1)根据字典的值进行排序
	key=operator.itemgetter(0)根据字典的键进行排序
	reverse降序排序字典
3、map函数  对可迭代数据进行逐个处理
4、counter函数 计数函数，注意返回格式  常用most_common
5、matplotlib图示汉字无法正常显示：
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
    mpl.rcParams['axes.unicode_minus']=False #用来正常显示负
6、
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_files

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from  sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import homogeneity_score,completeness_score,v_measure_score\
                            ,adjusted_rand_score,silhouette_score  (聚类)


from sklearn.externals.six import StringIO
from sklearn.externals import joblib

7、matplotlib
    plt.title(title)
    plt.legend
    plt.xlabel()
    plt.ylabel()

    plt.grid()
    plt.figure
    plt.subplot
    fig, axs = plt.subplots
    plt.fill_between
    plt.plot

    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']

    #绘制多个子窗口的程序示例
    fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'red')                        #绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'red')                        #绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'red')                        #绘制回归曲线
    axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)                #绘制样本点
    axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)                #绘制样本点
    axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)                #绘制样本点
    axs0_title_text=axs[0].set_title('k=1.0')
    axs1_title_text=axs[1].set_title('k=0.01')
    axs2_title_text=axs[2].set_title('k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

    #绘制一个窗口的程序实例
     fig = plt.figure()
     ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()

    #绘制多个叠加图层add_axes
    fig=plt.figure()
    rect=[0.1, 0.1, 0.8, 0.8]
    ax0 = fig.add_axes(rect, label='ax0',xticks=[],yticks=[])
    img=plt.imread('./Portland.png')
    ax0.imshow(img)
    # 叠加图层时frameon必须设置成False，不然会覆盖下面的图层
    ax1=fig.add_axes(rect, label='ax0',frameon=False)
    ax1.scatter(centers.A[:,0],centers.A[:,1],s=200,c='r',marker='+')
    plt.show()

    #调节多个子图之间的布局
    plt.figure(figsize=(2 * n_col, 2.2 * n_row), dpi=144)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.01)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')

8、把概率低的时间定义为1，positive
    查准率P 召回率R  F1score=2*p*r/(p+r)
9、KNN
    #返回训练样本x中n_neighbors距离最近的点,返回数据类型array
    neighbors=knn.kneighbors(x_test,return_distance=False)

10、numpy
    np.nonzero #在二维矩阵中的使用
    np.delete
    np.count_nonzero()
    np.tile()
    np.isnan
    np.cov

    import numpy as np
    a=np.mat(np.eye(2))
    b=np.array([1,-1]).T #维度不一样
    e=a.A[b==-1]

    b=np.mat([1,-1]).T #唯独不一样
    c=a[np.nonzero(b==-1)[0],:]
    d=a.A[np.nonzero(b==-1)[0]]

    dataMat[np.nonzero(clusterAssment[:,0].A==i)[0]]

11、列表删除操作 修改原列表 del
12、pickle
    写入 pickle.dump
    读取 pickle.load

13、pandas
    pd.Dataframe()

    dataframe:
    df.columns 返回列的名称
    df.keys 返回列的名称
    df['columns_name'] 取某列
    df.values 值取出
    df.drop 默认处理列 axis=1处理行
    df.fillna 填充缺失值

14、r'\w+'
15、合并zip  分开zip(*)
16、itertools.chain.from_iterable(a) 多重嵌套列表展开为一维 [[1,2],[3,4]] -> [1,2,3,4]
17、缺失值处理
    使用可用特征的均值来填补缺失值；
    使用特殊值来填补缺失值，如-1；
    忽略有缺失值的样本；
    使用相似样本的均值添补缺失值；
    使用另外的机器学习算法预测缺失值。
18、 class_weight有什么作用！！
    计算方法：n_samples / (n_classes * np.bincount(y))。
    n_samples为样本数，n_classes为类别数量，np.bincount(y)为每个类的样本数
    在分类模型中，我们经常会遇到两类问题：
    1.第一种是误分类的代价很高。比如对合法用户和非法用户进行分类，
    将非法用户分类为合法用户的代价很高，我们宁愿将合法用户分类为非法用户，
    这时可以人工再甄别，但是却不愿将非法用户分类为合法用户。
    这时，我们可以适当提高非法用户的权重。
    2. 第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，
    里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，则我们可以将所有的
    测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。
    这时，我们可以选择balanced，让类库自动提高非法用户样本的权重。
    提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，
    从而可以解决上面两类问题。
19、ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
    先打乱顺序，再划分
    划分成10个训练集、测试集的组合，且每个组合中train、test比例为0.8：0.2

    KFold 按序划分，划分成n_split个组合，且每个组合中测试集占1/n_split

20、learning_curve程序主要观察当训练数据集数目变化时，训练分数、测试分数的变化
    param_curve程序主要观察当模型参数发生变化时，训练分数、测试分数的变化
    二者都可以观察出模型的拟合情况
（**拟合情况与模型和数据集都有关系**）
    模型：模型的复杂程度、模型参数
    数据集：样本数、特征数、特征

21、分别执行两个语句，类似于Jupiter的操作
        #%%
        print(1)

        #%%
        print(2)

'''













