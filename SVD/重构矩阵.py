from numpy import linalg as la
from numpy import *

def loadExData():
    # 原矩阵
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


if __name__ == "__main__":
    # 对矩阵进行SVD分解(用python实现SVD)
    Data = loadExData()
    print('Data:', Data)
    U, Sigma, VT = linalg.svd(Data)
    # 打印Sigma的结果，因为前3个数值比其他的值大了很多，为9.72140007e+00，5.29397912e+00，6.84226362e-01
    # 后两个值比较小，每台机器输出结果可能有不同可以将这两个值去掉
    print('U:', U)
    print('Sigma', Sigma)
    print('VT:', VT)
    print('VT:', VT.T)

    # 重构一个3x3的矩阵Sig3
    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print(U[:, :3] * Sig3 * VT[:3, :])



