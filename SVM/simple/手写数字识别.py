import os
from sklearn.svm import SVC
import numpy as np

def img2vector(filepath):
    with open(filepath,'r+') as f:
        retVector=list(map(int,f.read().replace('\n','')))
    return retVector

def handwritingClassTest():
    train_digit='./digits/trainingDigits'
    files=os.listdir(train_digit)
    train_mat=[]
    label_mat=[]
    for file in files:
        label=file.split('_')[0]
        label_mat.append(label)

        vector=img2vector(os.path.join(train_digit,file))
        train_mat.append(vector)
    train_mat=np.array(train_mat)
    clf=SVC(C=100)
    clf.fit(train_mat,label_mat)

    test_digit='./digits/testDigits'
    files = os.listdir(test_digit)
    total=0.0
    error=0.0
    for file in files:
        total+=1
        label = file.split('_')[0]

        vector = img2vector(os.path.join(test_digit, file))
        vector=np.array(vector).reshape(1,-1)
        pred=clf.predict(vector)
        if pred!=label:
            error+=1
    print('测试错误率：{}/{}'.format(error,total))

if __name__ == '__main__':
    handwritingClassTest()