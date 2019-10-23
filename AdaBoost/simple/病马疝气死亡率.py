import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    traindata, trainlabel = loadDataSet('../horseColicTraining2.txt')
    testdata, testlabel = loadDataSet('../horseColicTest2.txt')

    clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    params = {'n_estimators':[1,2,3,4,5,6,7,8,9,10], #np.arange(1,10+1,1) x
              'algorithm':['SAMME','SAMME.R'],
              'base_estimator__max_depth':[1,2,3,4,5,6,7,8,9,10],
              'base_estimator__criterion':['gini','entropy']
              }
    model=GridSearchCV(clf,params,cv=10)
    model.fit(traindata,trainlabel)
    print('最优参数：',model.best_params_)
    print('训练分数：',model.best_score_)

    clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=10,algorithm='SAMME')
    clf.fit(traindata,trainlabel)
    test=clf.predict_proba(testdata)
    train_score=clf.score(traindata,trainlabel)
    test_score=clf.score(testdata,testlabel)
    print('训练分数：{:.5f}，测试分数：{:.5f}'.format(train_score,test_score))

    import scikitplot as skplt
    import matplotlib.pyplot as plt
    skplt.metrics.plot_roc(testlabel,test)
    plt.show()