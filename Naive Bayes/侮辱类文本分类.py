import numpy as np
from math import log

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]   																#类别标签向量，1代表侮辱性词汇，0代表不是
	return postingList,classVec

def createVocabList(dataSet):
	retVector=[]
	for sublist in dataSet:
		retVector.extend(sublist)
	return list(set(retVector))

#词集模型
def setOfWords2Vec(vocabList, inputSet):
	retVector=[0]*len(vocabList)
	for i in inputSet:
		if i in vocabList:
			retVector[vocabList.index(i)]=1
	return retVector

#词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    retVec=[0]*len(vocabList)
    for i in inputSet:
        if i in vocabList:
            retVec[vocabList.index(i)]+=1
    return retVec

#伯努利模型 使用词集模型获得trainmat
def trainNB0(trainMatrix, trainCategory):
	numdoc=len(trainMatrix)
	numwords=len(trainMatrix[0])
	p1=float(sum(trainCategory))/float(numdoc) #侮辱类文档占比

	#避免出现条件概率为0的情况，并采用log避免下溢出
	#拉普拉斯平滑
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

#多项式模型 使用词袋模型获得trainmat
def MBtrainNB0(trainMatrix, trainCategory):
	numdoc=len(trainMatrix)
	numwords=len(trainMatrix[0])
	p1=float(trainMatrix[np.nonzero(trainCategory[:]==1),:].sum())/float(trainMatrix.sum()) #侮辱类文档占比

	#避免出现条件概率为0的情况，并采用log避免下溢出
	#拉普拉斯平滑
	p1num_c=np.ones(numwords)
	p0num_c=np.ones(numwords)
	p1num=2.0
	p0num=2.0
	for i in range(numdoc):
		if trainCategory[i]==1:
			p1num_c+=trainMatrix[i]
			p1num+=sum(trainMatrix[i])
		elif trainCategory[i]==0:
			p0num_c+=trainMatrix[i]
			p0num+=sum(trainMatrix[i])
	p1vec=np.log(p1num_c/p1num)
	p0vec=np.log(p0num_c/p0num)

	return p1vec,p0vec,p1

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1=np.dot(vec2Classify,p1Vec)+log(pClass1)
	#np.dot 一维向量相乘在相加，即求内积
	p0=np.dot(vec2Classify,p0Vec)+log(1-pClass1)
	if p1>p0:
		return 1
	else:
		return 0

def testingNB():
	dataset, classVec = loadDataSet()
	vocabList = createVocabList(dataset)

	#以下是伯努利模型计算方法
	trainmat=[]
	for i in dataset:
		trainmat.append(setOfWords2Vec(vocabList,i))
	p1vec, p0vec, p1=trainNB0(np.array(trainmat),np.array(classVec))

	test_words=['stupid', 'garbage', 'dalmation']
	test_vec=setOfWords2Vec(vocabList,test_words)
	result=classifyNB(np.array(test_vec),p0vec,p1vec,p1)
	if result:
		print('侮辱类')
	else:
		print('非侮辱类')

	#以下是多项式模型
	# trainmat=[]
	# for i in dataset:
	# 	trainmat.append(bagOfWords2VecMN(vocabList,i))
	# p1vec, p0vec, p1=MBtrainNB0(np.array(trainmat),np.array(classVec))
	#
	# test_words=['stupid', 'garbage', 'dalmation']
	# test_vec=bagOfWords2VecMN(vocabList,test_words)
	# result=classifyNB(np.array(test_vec),p0vec,p1vec,p1)
	# if result:
	# 	print('侮辱类')
	# else:
	# 	print('非侮辱类')

if __name__ == '__main__':
	testingNB()
