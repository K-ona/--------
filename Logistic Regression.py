import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

class DimensionValueError(ValueError):
    pass

class TypeError(ValueError):
    pass

class IterError(ValueError):
    pass

class DataProcess:
    # 读入数据
    def loadCSV(self, path):
        f = csv.reader(open(path))
        l = []
        for row in f:
            l.append(row)
        l = np.array(l[1:], dtype = np.float64)
        return l

    #预处理
    def Predata(self, path): 
        csvf = self.loadCSV(path)
        data = csvf[:, :2]
        labels = csvf[:, 2:]
        
        # labels.resize((labels.shape[0], ))
        labels = labels.reshape((labels.shape[0]))
        return data, labels

    #划分数据集
    def Split_Data(self, path, t_s=0.3):
        data, labels = self.Predata(path)
        #为w和b和统一表示，加一维全为1的值
        # data = np.hstack((np.ones((data.shape[0], 1)), data))
        return train_test_split(data, labels, test_size = t_s, random_state = 0)

class Logistic:
    #初始化
    def __init__(self, data, labels, gd, learning_rate = 0.1, n_iter = 3000):

        if data.shape[0] != labels.shape[0]:
            raise DimensionValueError("Dimension Error")
        
        if not isinstance(gd, str):
            raise TypeError("Type Error")

        data = np.hstack((np.ones((data.shape[0], 1)), data))
        self.data = data
        self.labels = labels
        self.gd = gd
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.init_weight()
  
   #sigmoid函数
    def sigmoid(self, x):  
        return 1 / (1 + np.exp(-x))

    #随机初始化w和b
    def init_weight(self):
        #w' = [b;w]  w和b统一为一个向量
        w = np.random.uniform(-1, 1, (self.data.shape[-1]))
        self.w = w
    
    # #计算梯度
    # def Gradient(self, index):
    #     print(self.data[index, :])
    #     s = self.w.dot(self.data[index, :])
    #     sig = self.sigmoid(s) - self.labels[index]
    #     res = sig.dot(self.data[index])
    #     return res

    #几种不同的梯度下降算法
    def AdaGrad(self):
        pass

    def VanillaGD(self):
        #N个M维的instance
        N = self.data.shape[0]
        M = self.data.shape[1]
        #计算梯度
        Del = self.sigmoid(self.w.T.dot(self.data.T))
        Sep = Del - self.labels
        Gradient = 1 / N * np.matmul(Sep, self.data)
        #更新参数

        self.w -= self.learning_rate * Gradient

    def Adam(self, x):
        pass
        
    def fit(self):
        for i in tqdm(range(self.n_iter)):
            getattr(self, self.gd)()
        
    #判别规则 >0.5为1，否则为0
    def Dicision(self, p):
        return np.around(self.sigmoid(p))

    def predict(self, data, labels):
        data = np.hstack((np.ones((data.shape[0], 1)), data))
        P = np.matmul(data, self.w)
        res = self.Dicision(P)
        res = res[res == labels]
        
        print('Error Rate: ', 1 - res.shape[0] / labels.shape[0])
        self.visual(data, labels, self.w)

    def visual(self, data, labels, weight):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        testdata_positive = data[labels == 1]
        testdata_negative = data[labels == 0]
        
        ax.scatter(testdata_positive[:, 1], testdata_positive[:, 2], c='red', label = '1')
        ax.scatter(testdata_negative[:, 1], testdata_negative[:, 2], c='black', label = '0')
        ax.plot(np.arange(0, 1.5),(-np.arange(0, 1.5)*weight[1]-weight[0])/weight[2])

        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.legend(loc=(1, 0))
        plt.show()



if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('data.csv')

    Train_data = np.array(Train_data, dtype=np.float64)
    Validation_data = np.array(Validation_data, dtype=np.float64)
    Train_labels = np.array(Train_labels, dtype=np.float64)
    Validation_labels = np.array(Validation_labels, dtype=np.float64)
    
    Log = Logistic(Train_data, Train_labels, 'VanillaGD')
    Log.fit()
    # Log.predict(Validation_data, Validation_labels)
    Log.predict(Train_data, Train_labels)

