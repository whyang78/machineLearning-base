import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np

if __name__ == '__main__':
    with open('./lenses.txt','r+') as f:
        content=f.readlines()
    lensesTarget=[line.strip().split('\t')[-1] for line in content]
    lenses=[line.strip().split('\t')[:-1] for line in content]

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    df=pd.DataFrame(lenses,columns=lensesLabels)

    test_vec =np.array([['presbyopic', 'myope', 'yes', 'normal']])
    le=LabelEncoder()
    for index,col in enumerate(df.columns):
        df[col]=le.fit_transform(df[col])
        test_vec[:,index]=le.transform(test_vec[:,index])

    myTree=DecisionTreeClassifier(max_depth=4,criterion='entropy')
    myTree.fit(df.values,lensesTarget)

    #绘制决策树固定格式代码
    dot_data = StringIO()
    export_graphviz(myTree, out_file=dot_data,  # 绘制决策树
                         feature_names=df.keys(),
                         class_names=myTree.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("lenses_tree.pdf")

    print(myTree.predict(test_vec))