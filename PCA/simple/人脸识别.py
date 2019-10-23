import numpy as np
import matplotlib.pyplot as plt
'''
1、加载数据
'''
from sklearn.datasets import fetch_olivetti_faces
faces=fetch_olivetti_faces('./',download_if_missing=False)

x=faces.data
y=faces.target
targets=np.unique(y)
target_names=np.array(['c%d'% d for d in targets])
n_target=target_names.shape[0]
n_sample,h,w=faces.images.shape
print('样本个数：{}，类别数目：{}'.format(n_sample,n_target))
print('图片大小：{}x{}'.format(h,w))
print('样本大小：{}'.format(x.shape))

def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
    plt.figure(figsize=(2*n_col,2.2*n_row))
    plt.subplots_adjust(left=0.01,bottom=0,right=0.99,top=0.9,hspace=0.01)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape(h,w),cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')

#显示12种类型图片
sample_images=None
sample_titles=[]
for i in range(n_target):
    image=x[y==i]
    index=np.random.randint(0,image.shape[0],1)
    image=image[index,:]
    if sample_images is not None:
        sample_images=np.concatenate((sample_images,image),axis=0)
    else:
        sample_images=image
    sample_titles.append(target_names[i])

n_col=6
n_row=2
plot_gallery(sample_images,sample_titles,h,w,n_row,n_col)
plt.show()

'''
2、划分数据集，使用SVC模型训练，查看训练结果
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=78)

from sklearn.svm import SVC
clf=SVC(class_weight='balanced')
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred,labels=range(n_target))
plt.matshow(cm, fignum=1, cmap='Reds')
plt.colorbar()
plt.show()

report=classification_report(y_test,y_pred)
print(report)
#效果很差，原因在于数据集是400 x 4096，特征数远大于样本数，过拟合现象很严重

'''
3、降维处理，选择合适的维度，保证较高的数据还原率
'''
from sklearn.decomposition import PCA

#绘制维度与数据还原率曲线
n_components_list=range(10,300,30)
explained_ratios = []
for c in n_components_list:
    pca=PCA(n_components=c)
    pca.fit_transform(x)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))

plt.figure()
plt.grid()
plt.plot(n_components_list,explained_ratios)
plt.title('explained_ratios with different n_components')
plt.xlabel('n_components')
plt.ylabel('explained_ratios')
plt.xticks(np.arange(0,300,20))
plt.yticks(np.arange(0.5,1.01,0.05))
plt.show()
#[140, 75, 37, 19, 8] 对应还原率 0.95 0.90 0.80 0.70 0.60

def title_prefix(prefix, title):
    return "{}: {}".format(prefix, title)

#展示不同数据还原率情况下对应的图片
sample_images=sample_images[:5]
sample_titles=sample_titles[:5]
n_components_list=[140, 75, 37, 19, 8]
plot_images=sample_images
plot_titles=[title_prefix('orig',t) for t in sample_titles]

for c in n_components_list:
    pca=PCA(n_components=c)
    pca.fit(x)

    sample_pca=pca.transform(sample_images)
    sample_pca_inv=pca.inverse_transform(sample_pca)
    plot_images=np.concatenate((plot_images,sample_pca_inv),axis=0)

    sample_title_pca=[title_prefix(str(c),t) for t in sample_titles]
    plot_titles=np.concatenate((plot_titles,sample_title_pca),axis=0)

plot_gallery(plot_images,plot_titles,h,w,len(n_components_list)+1,5)
plt.show()

'''
4、选择较好的数据还原率对应的维度先进行降维处理，然后再使用SVC模型，寻找最优参数，
    最终查看训练结果
'''
#先进行降维处理
k=140
pca=PCA(n_components=k,svd_solver='randomized', whiten=True)
pca.fit(x_train)
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)

#最优参数选择
from sklearn.model_selection import GridSearchCV
param_grid={'C':[1,5,10,50,100],
            'gamma':[0.0001,0.0005,0.001,0.005,0.01]}
model=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid,cv=5)
model.fit(x_train_pca,y_train)
print('最优参数：{}'.format(model.best_params_))

#最优模型测试，查看训练结果
y_pred=model.best_estimator_.predict(x_test_pca)
cm=confusion_matrix(y_test,y_pred)
plt.matshow(cm, fignum=1, cmap='Reds')
plt.colorbar()
plt.show()

report=classification_report(y_test,y_pred)
print(report)




