import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

# 定义异常类
class DimensionValueError(ValueError):
    pass

# 读入数据
def loadArffData(path):
    data = arff.loadarff(path) 
    df = pd.DataFrame(data[0]) 
    return df

def Predata(path):
    df = loadArffData(path)
    labels = df.values[:, -1]
    labels = labels == b'tested_positive'
    
    data = df.values[:, 0:df.values.shape[-1] - 1].T
    return data, labels

def datasplit(data, labels):

    tmp = np.hstack((labels.reshape((-1, 1)), data))
    df = pd.DataFrame(tmp, columns=['lable', 'x', 'y'])

    data1 = df.loc[df.lable == True]
    data2 = df.loc[df.lable == False]
    print(data1)
    print(data2)    
    return data1, data2

#图示
def showInImg(data, labels):

    data1, data2 = datasplit(data, labels)
    # data1 = pd.DataFrame(data1, columns=['x', 'y'])
    # data2 = pd.DataFrame(data2, columns=['x', 'y'])
    
    #分别画出scatter图，但是设置不同的颜色
    plt.scatter(data1['x'], data1['y'], color='blue', label='negative')
    plt.scatter(data2['x'], data2['y'], color='green', label='positive')

    #设置图例
    plt.legend(loc=(1, 0))

    #显示图片
    plt.show()


# 直接计算

class PCA(object):

    def __init__(self, x,  d_dimension = 2):
        self.x = x
        self.dimension = x.shape[0]

        if d_dimension > self.dimension:
            raise DimensionValueError("dimension error")

        self.ddimension = d_dimension

    def Covariance(self):
        return np.cov(self.x.astype(np.float32)) #一条记录为一行
        # return np.cov(self.x.T) #一条记录为一列

    def SortEig(self):
        cov = self.Covariance()
        eigvalue, eigvector = np.linalg.eig(cov)
        eigvector = np.transpose(eigvector)
        tmp = np.hstack((eigvalue.reshape((eigvalue.shape[0], 1)), eigvector))
        tmp = pd.DataFrame(tmp)
        res = tmp.sort_values(by=0, ascending=False)
        return res

    def dimensionReductionBydimension(self):
        P = self.SortEig().values[:self.ddimension, 1:]

        res = P.dot(self.x)
        return res.T
        

# 0均值化后降维

def pca(input_x, d_dimension = 2):
    row, column = input_x.shape
    
    #0均值化
    mean = np.mean(input_x, axis=-1)
    mean = np.array(list(mean) * column).reshape(column, row).T 
    input_x = input_x.astype(np.float64) - mean  

    Covariance = input_x.dot(input_x.T)
    # print(Covariance.shape)
    # Covariance = np.cov(np.transpose(input_x))

    eigvalue, eigvector = np.linalg.eig(Covariance)
    eigvector = np.transpose(eigvector)
    tmp = np.hstack((eigvalue.reshape((eigvalue.shape[0], 1)), eigvector))
    tmp = pd.DataFrame(tmp)
    res = tmp.sort_values(by=0, ascending=False)

    P = res.values[:d_dimension, 1:]
    result = P.dot(input_x)
    return result.T


if __name__ == "__main__":

    data, labels = Predata('diabetes.arff')

    # print(pca(input_x=data,d_dimension = 2))
    showInImg(pca(input_x=data,d_dimension = 2), labels)

    x = PCA(x=data, d_dimension=2)
    # print(x.dimensionReductionBydimension())
    showInImg(x.dimensionReductionBydimension(), labels)
