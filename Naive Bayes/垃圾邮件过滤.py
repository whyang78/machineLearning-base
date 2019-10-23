import numpy as np
from math import log
import nltk
from nltk.corpus import stopwords
import random

def createVocabList(dataSet):
    retVec=[]
    for i in dataSet:
        retVec.extend(i)
    return list(set(retVec))

def setOfWords2Vec(vocabList, inputSet):
    retVec=[0]*len(vocabList)
    for i in inputSet:
        if i in vocabList:
            retVec[vocabList.index(i)]=1
    return retVec

def bagOfWords2VecMN(vocabList, inputSet):
    retVec=[0]*len(vocabList)
    for i in inputSet:
        if i in vocabList:
            retVec[vocabList.index(i)]+=1
    return retVec

def trainNB0(trainMatrix, trainCategory):
    numdoc=len(trainMatrix)
    numwords=len(trainMatrix[0])
    p1=float(sum(trainCategory))/float(numdoc)
    p1num_c=np.ones(numwords)
    p0num_c=np.ones(numwords)
    p1num=2.0
    p0num=2.0

    for i in range(numdoc):
        if trainCategory[i]==1:
            p1num_c+=trainMatrix[i]
            p1num+=1
        elif trainCategory[i]==0:
            p0num_c+=trainMatrix[i]
            p0num+=1
    p1vec=np.log(p1num_c/p1num)
    p0vec=np.log(p0num_c/p0num)
    return p1vec,p0vec,p1

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1=np.dot(vec2Classify,p1Vec)+log(pClass1)
    p0=np.dot(vec2Classify,p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def textParse(bigString):
    text=bigString.lower()
    words=nltk.regexp_tokenize(text,r'\w+')
    new_words=[]
    for i in words:
        if i not in stopwords.words('english'):
            new_words.append(i)
    return new_words


def spamTest():
    docList = []
    classList = []
    for i in range(1, 26):                                                  #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())     #读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)                                                 #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())      #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)                                    #创建词汇表，不重复
    indexList=list(range(50))
    random.shuffle(indexList)

    trainMat = []
    trainClasses = []
    for i in indexList[:-10]:
        trainMat.append(setOfWords2Vec(vocabList,docList[i]))
        trainClasses.append(classList[i])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  #训练朴素贝叶斯模型

    errorCount = 0                                                          #错误分类计数
    for i in indexList[-10:]:
        wordVector = setOfWords2Vec(vocabList, docList[i])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[i]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[i])
    print('错误率：%.2f%%' % (float(errorCount) / 10 * 100))


if __name__ == '__main__':
    spamTest()