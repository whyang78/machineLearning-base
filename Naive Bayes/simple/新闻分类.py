import matplotlib.pyplot as plt
import jieba
import os
import random
import itertools
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
random.seed(78)
def TextProcessing(folder_path, test_size = 0.2):
    dirs=os.listdir(folder_path)
    wordList=[]
    classList=[]
    for fold in dirs:
        file_path=os.path.join(folder_path,fold)
        files=os.listdir(file_path)
        for file in files:
            with open(os.path.join(file_path,file),'r+',encoding='utf-8') as f:
                content=f.read()
            words=jieba.cut(content,cut_all=False)
            words=list(words)
            wordList.append(words)
            classList.append(fold)

    word_class_list=list(zip(wordList,classList))
    random.shuffle(word_class_list)
    index=int(len(word_class_list)*test_size)
    test_dataset=word_class_list[:index]
    train_dataset=word_class_list[index:]
    train_words,train_classes=zip(*train_dataset)
    test_words,test_classes=zip(*test_dataset)

    all_words=list(itertools.chain.from_iterable(train_words))
    all_words_count=Counter(all_words)
    count_list = sorted(all_words_count.items(), key=lambda x: x[1], reverse=True)
    all_words_list,all_words_num=zip(*count_list)
    all_words_list=list(all_words_list)

    return all_words_list,train_words,train_classes,test_words,test_classes
def MakeWordsSet(words_file):
    wordsSet=set()
    with open(words_file,'r+',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.strip()
            if len(line)>0:
                wordsSet.add(line)
    return wordsSet

def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words_list=[]
    n=1
    for i in range(deleteN,len(all_words_list),1):
        if n>1000:
            break
        if not all_words_list[i].isdigit() and all_words_list[i] not in stopwords_set and 1<len(all_words_list[i])<5:
            feature_words_list.append(all_words_list[i])
            n+=1

    return feature_words_list

def TextFeatures(train_data_list, test_data_list, feature_words):
    def textFeat(text,feature):
        #词袋 MB用此种准确率稍高一些
        # feat=[0]*len(feature)
        # for word in text:
        #     if word in feature:
        #         feat[feature.index(word)]+=1
        text=set(text)
        feat=[1 if word in text else 0 for word in feature]
        return feat

    train_feature_data=[textFeat(text,feature_words) for text in train_data_list]
    test_feature_data=[textFeat(text,feature_words) for text in test_data_list]
    return train_feature_data,test_feature_data

if __name__ == '__main__':

    folder_path = './SogouC/Sample'  # 训练集存放地址
    all_words_list, train_data_list, train_class_list,test_data_list, test_class_list = TextProcessing(folder_path,
                                                                                                        test_size=0.2)
    # 生成stopwords_set
    stopwords_file = './SogouC/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)  # 0 20 40 60 ... 980
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        clf=MultinomialNB()
        clf.fit(train_feature_list,train_class_list)
        test_accuracy =clf.score(test_feature_list,test_class_list)
        test_accuracy_list.append(test_accuracy)

    avr=lambda x:sum(x)/len(x)
    print(avr(test_accuracy_list))

    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()


