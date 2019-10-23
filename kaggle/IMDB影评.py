'''
1、加载数据集，数据预处理
'''
import pandas as pd
train=pd.read_csv('./Datasets/IMDB/labeledTrainData.tsv',delimiter='\t')
test=pd.read_csv('./Datasets/IMDB/testData.tsv',delimiter='\t')
# print(train.head())
# print(train.columns) 'id', 'sentiment', 'review'
# print(test.head())

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

def review_to_text(review,remove_stopwords):
    #去掉html印记
    content=BeautifulSoup(review,'html').get_text()
    #去掉非字母字符
    letters=re.sub('[^a-zA-Z]',' ',content)
    words=letters.lower().split()
    #是否去除停用词
    if remove_stopwords:
        stops=set(stopwords.words('english'))
        words=[w for w in words if w not in stops]
    return words

x_train=[]
for review in train['review']:
    x_train.append(' '.join(review_to_text(review,True)))

x_test=[]
for review in test['review']:
    x_test.append(' '.join(review_to_text(review,True)))

y_train=train['sentiment']

'''
2、特征抽取，模型训练，参数优化
   对比两种特征抽取方法
   词频编码+多项式型贝叶斯
   tfidf编码+多项式型贝叶斯
'''
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

pip_count=Pipeline([('count_vec',CountVectorizer(analyzer='word')),
                    ('mnb',MultinomialNB())])
pip_tfidf=Pipeline([('tfidf_vec',TfidfVectorizer(analyzer='word')),
                    ('mnb',MultinomialNB())])

# CountVectorizer(ngram_range=(1,1),binary=False,)
'''
ngram_range:要提取的n-gram的n-values的下限和上限范围，
            在min_n <= n <= max_n区间的n的全部值。
            可理解为某个词与前面几个词有关。（朴素贝叶斯理论上无关，但是可以设置这个考虑有关）
binary：如果False，所有非零计数被设置为1，这对于离散概率模型是有用的，
        建立二元事件模型，而不是整型计数。可理解为True为词袋模型，
        False为词集模型。
'''
params_count={'count_vec__binary':[True,False],
              'count_vec__ngram_range':[(1,1),(1,2)],
              'mnb__alpha':[0.1,1.0,10.0]}
params_tfidf={'tfidf_vec__binary':[True,False],
              'tfidf_vec__ngram_range':[(1,1),(1,2)],
              'mnb__alpha':[0.1,1.0,10.0]}

gs_count=GridSearchCV(pip_count,params_count,cv=4,n_jobs=-1)
gs_count.fit(x_train,y_train)
print(gs_count.best_score_)
print(gs_count.best_params_)
count_y_test=gs_count.best_estimator_.predict(x_test)
submission_count=pd.DataFrame({'id':test['id'],'sentiment':count_y_test})
submission_count.to_csv('./submission_count.csv',index=False)

gs_tfidf=GridSearchCV(pip_tfidf,params_tfidf,cv=4,n_jobs=-1)
gs_tfidf.fit(x_train,y_train)
print(gs_tfidf.best_score_)
print(gs_tfidf.best_params_)
tfidf_y_test=gs_tfidf.best_estimator_.predict(x_test)
submission_tfidf=pd.DataFrame({'id':test['id'],'sentiment':tfidf_y_test})
submission_tfidf.to_csv('./submission_tfidf.csv',index=False)

'''
3、利用没标记训练集进行词向量word2vec,训练词向量模型
'''
unlabeled_train=pd.read_csv('./Datasets/IMDB/unlabeledTrainData.tsv',delimiter='\t',
                            quoting=3)

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review,tokenizer):
    all_sentences=tokenizer.tokenize(review.strip())
    sentences=[]
    for sub in all_sentences:
        if len(sub)>0:
            sentences.append(review_to_text(sub,False))
    return sentences

corpora=[]
for review in unlabeled_train['review']:
    corpora+=review_to_sentences(review,tokenizer)

#设置词向量模型参数
num_features = 300
min_word_count = 20
context = 10
downsampling = 1e-3

from gensim.models import word2vec
model=word2vec.Word2Vec(corpora,size=num_features,min_count=min_word_count,
                        sample=downsampling,window=context)

model.init_sims(replace=True)
#保存模型
model_path='./word2vec_model'
model.save(model_path)

from gensim.models import Word2Vec
#载入模型
model=Word2Vec.load(model_path)

# model.most_similar('man') 检验一下模型

'''
4、利用词向量模型将数据集转换为向量模式，
    并采用GBDT模型进行训练，参数优化
'''
import numpy as np

def makeFeatureVec(words, model, num_features):
    feat=np.zeros((num_features),dtype='float32')
    index2word=set(model.wv.index2word)
    count_words=0
    for word in words:
        if word in index2word:
            count_words+=1
            feat=np.add(feat,model[word])
    feat=np.divide(feat,count_words) #平均词向量
    return feat

def getAvgFeatureVecs(reviews, model, num_features):
    feat=np.zeros((len(reviews),num_features),dtype='float32')

    index=0
    for review in reviews:
        feat[index]=makeFeatureVec(review,model,num_features)
        index+=1
    return feat

#将训练集和测试集每个review转换成词向量
new_train=[]
for review in train['review']:
    new_train.append(review_to_text(review,True))
trainVec=getAvgFeatureVecs(new_train,model,num_features)

new_test=[]
for review in test['review']:
    new_test.append(review_to_text(review,True))
testVec=getAvgFeatureVecs(new_test,model,num_features)

#模型训练,参数优化
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

gbc=GradientBoostingClassifier()
params_gbc={'learning_rate':[0.01,0.1,1.0],
            'n_estimators':[10,500,100],
            'max_depth':[2,3,4]}
gs=GridSearchCV(gbc,params_gbc,cv=4,verbose=1)
gs.fit(trainVec,y_train)
print(gs.best_score_)
print(gs.best_params_)

gbc_y_predict=gs.best_estimator_.predict(testVec)
submission_gbc=pd.DataFrame({'id':test['id'],'sentiment':gbc_y_predict})
submission_gbc.to_csv('./submission_gbc.csv',index=False,quoting=3)

