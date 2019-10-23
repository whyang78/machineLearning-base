import numpy as np

def loadDataSet(fileName, delim='\t'):
    data=[]
    with open(fileName,'r+') as f:
        lines=f.readlines()
        line_arr=[line.strip().split(delim) for line in lines]
        data=[list(map(float,line)) for line in line_arr]
    return np.mat(data)

def replaceNanWithMean():
    datamat=loadDataSet('./secom.data',' ')
    numFeat=datamat.shape[1]
    for i in range(numFeat):
        datamat_NoNan_i=datamat[np.nonzero(~np.isnan(datamat[:,i].A))[0],i]
        mean=np.mean(datamat_NoNan_i,axis=0)
        datamat[np.nonzero(np.isnan(datamat[:,i].A))[0],i]=mean
    return datamat

def analyse_data(dataMat):
    mean=np.mean(dataMat,axis=0)
    new_datamat=dataMat-mean
    cov_datamat=np.cov(new_datamat,rowvar=False)

    topFeat=20
    #特征向量是一个一个列向量
    eigVal,eigVect=np.linalg.eig(cov_datamat)
    sort_index=np.argsort(eigVal)[:-(topFeat+1):-1]
    # new_eigVect=eigVect[:,sort_index]

    cov_all_var=float(np.sum(eigVal))
    sum_var=0.0

    for i in range(len(sort_index)):
        var=float(eigVal[sort_index[i]])
        sum_var+=var
        print('主成分：{},方差占比：{:.4f},累计方差占比:{:.4f}'.format(i+1,var/cov_all_var,sum_var/cov_all_var))

if __name__ == '__main__':
    datamat=replaceNanWithMean()
    analyse_data(datamat)