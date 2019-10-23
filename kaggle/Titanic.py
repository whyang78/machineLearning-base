'''
1、加载数据集，数据预处理
'''
import pandas as pd
train=pd.read_csv('./Datasets/Titanic/train.csv')
test=pd.read_csv('./Datasets/Titanic/test.csv')
# print(train.info())
# print(test.info())

#去除无用特征以及过度缺失特征
selected_feature=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x_train=train[selected_feature]
x_test=test[selected_feature]
y_train=train['Survived']

#填充缺失值
# print(x_train.isna().sum())
# print(x_test.isna().sum())

#对于Embarked这种类别型特征，可以使用出现频率最高的特征值进行填充
# print(x_train['Embarked'].value_counts())
x_train['Embarked'].fillna('S',inplace=True)

#对于age这种数值型特征，可以使用求平均值或者中位数进行填充
x_train['Age'].fillna(x_train['Age'].mean(),inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(),inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(),inplace=True)
# print(x_train.info())
# print(x_test.info())

'''
2、标签数据处理 两种方式 ovk 增量型
'''
from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
x_train=dict_vec.fit_transform(x_train.to_dict(orient='record'))
x_test=dict_vec.transform(x_test.to_dict(orient='record'))

# x_train['Sex']=(x_train['Sex']=='male').astype('int')
# x_test['Sex']=(x_test['Sex']=='male').astype('int')
# all_Embarked=x_train['Embarked'].unique().tolist()
# x_train['Embarked']=x_train['Embarked'].apply(lambda x:all_Embarked.index(x))
# x_test['Embarked']=x_test['Embarked'].apply(lambda x:all_Embarked.index(x))

'''
3、模型训练(随机森林、xgboost)，交叉验证，学习曲线绘制，
    生成csv文件
'''
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
rfc=RandomForestClassifier()
xgbc=XGBClassifier()

from sklearn.model_selection import cross_val_score,ShuffleSplit
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=78)
rfc_scores=cross_val_score(rfc,x_train,y_train,cv=cv)
xgbc_scores=cross_val_score(xgbc,x_train,y_train,cv=cv)

#绘制学习曲线查看拟合情况
from learning_curve import plot_learning_curve
import matplotlib.pyplot as plt
plt.figure(figsize=(18,6))
plt.subplot(121)
plot_learning_curve(xgbc,'xgbc',x_train,y_train,cv=cv)
plt.subplot(122)
plot_learning_curve(rfc,'rfc',x_train,y_train,cv=cv)
plt.show()

rfc.fit(x_train,y_train)
rfc_y_test=rfc.predict(x_test)
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_test})
rfc_submission.to_csv('./rfc_submission.csv',index=False)

xgbc.fit(x_train,y_train)
xgbc_y_test=xgbc.predict(x_test)
xgbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':xgbc_y_test})
xgbc_submission.to_csv('./xgbc_submission.csv',index=False)

'''
4、参数优化，提升模型性能(xgboost)
'''
from sklearn.model_selection import GridSearchCV
params={'max_depth':range(2,7),
        'learning_rate':[0.01,0.05,0.1,0.5,1.0],
        'n_estimators':range(20,200,20)}
clf=GridSearchCV(xgbc,params,cv=5,n_jobs=-1)
clf.fit(x_train,y_train)

best_xgbc=clf.best_estimator_
best_xgbc_y_predict=best_xgbc.predict(x_test)
best_xgbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':best_xgbc_y_predict})
best_xgbc_submission.to_csv('./best_xgbc_submission.csv',index=False)