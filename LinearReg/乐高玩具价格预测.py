from bs4 import BeautifulSoup
import numpy as np
import random

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99

def regularize(xMat, yMat):
    x_mean=np.mean(xMat,axis=0)
    x_var=np.var(xMat,axis=0)
    x=(xMat-x_mean)/x_var

    y_mean=np.mean(yMat,axis=0)
    y=yMat-y_mean
    return x,y

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

def standRegres(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T

    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        return
    w=xTx.I*xMat.T*yMat
    return w

def ridgeRegres(xMat, yMat, lam = 0.2):
    m,n=xMat.shape
    temp=xMat.T*xMat+np.mat(np.eye(n))*lam
    if np.linalg.det(temp)==0.0:
        return
    w=temp.I*xMat.T*yMat
    return w
def ridgeTest(xArr, yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T

    #数据标准化
    y_mean=np.mean(yMat,axis=0)
    yMat=yMat-y_mean
    x_mean=np.mean(xMat,axis=0)
    x_var=np.var(xMat,axis=0)
    xMat=(xMat-x_mean)/x_var

    numItem=30
    wMat=np.zeros((numItem,xMat.shape[1]))
    for i in range(30):
        w=ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=w.T

    return wMat

def useStandRegres():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)

    lgX=np.mat(lgX)
    m,n=lgX.shape
    col=np.mat(np.ones((m,1)))
    lgX=np.concatenate((col,lgX),axis=1)

    w=standRegres(lgX,lgY)
    print('常值:',w[0])
    print('年份:',w[1])
    print('部件数量:',w[2])
    print('是否为全新:',w[3])
    print('原价:',w[4])

def crossValidation(xArr, yArr, numVal = 10):
    xMat=np.array(xArr)
    yMat=np.array(yArr)

    m,n=xMat.shape
    indexList=list(range(m))
    errorMat=np.mat(np.zeros((numVal,30)))
    random.shuffle(indexList)
    for i in range(numVal):
        trainX=xMat[indexList[:int(0.9*m)]]
        trainY=yMat[indexList[:int(0.9*m)]]
        testX=xMat[indexList[int(0.9*m):]]
        testY=yMat[indexList[int(0.9*m):]]

        wMat=ridgeTest(trainX,trainY)
        for j in range(30):
            testX=np.mat(testX)
            testY=np.mat(testY)

            #使用训练集的均值和方差
            mean_x=np.mean(trainX,0)
            var_x=np.var(trainX,0)
            testX=(testX-mean_x)/var_x
            mean_y=np.mean(trainY,0)

            pred_y=testX*np.mat(wMat)[j,:].T+mean_y
            errorMat[i,j]=rssError(pred_y.A,testY.A)

    mean_error=np.mean(errorMat,0)
    min_error=float(np.min(mean_error))
    best_weight=wMat[mean_error.A[0]==min_error]
    return best_weight


if __name__ == '__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    bestweight=crossValidation(lgX,lgY)