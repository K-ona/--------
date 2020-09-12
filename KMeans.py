from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

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

class KMeans():
    def __init__(self):
        pass

    def generate_krandom_center(self, data):
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        
        self.center = np.random.randn(self.k, data.shape[1]) * std + mean
        pass
    
    def cal_dis(self, data): 
        for i in range(self.k): 
            self.dis[:, i] = np.linalg.norm(data - self.center[i], axis=1)
        
        #得到最小值的下标
        self.clusters = np.argmin(self.dis, axis=1)
        #重新计算中心点
        for i in range(self.k): 
            self.center[i] = np.mean(data[self.clusters == i], axis=0)
        pass

    def fit(self, data, k = 3, epoch = 500, eps = 1e-8): 
        self.k = k
        self.generate_krandom_center(data)
        self.dis = np.zeros((data.shape[0], self.k))
        
        for i in tqdm(range(epoch)): 
            pre_center = deepcopy(self.center)
            self.cal_dis(data) 
            sep = np.linalg.norm(pre_center - self.center)
            loss = self.loss(data) 
            print('sep = ', sep, 'loss = ', loss)
            if sep < eps or loss < .5:
                break
        self.visual(data)
        return self

    def visual(self, data): 
        plt.clf()
        plt.scatter()
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c=self.clusters)
        plt.scatter(self.center[:, 0], self.center[:, 1], marker='*', c='k')
        plt.show()
    
    def predict(self, data):
        self.dis = np.zeros((data.shape[0], self.k))
        self.cal_dis(data)
        loss = self.loss(data)
        print('loss = ', loss)
        self.visual(data)

    def loss(self, data): 
        loss = .0
        # self.visual(data)
        for i in range(self.k):
            loss += np.sum(np.linalg.norm(data[self.clusters == i] - self.center[i]))
        # print('loss = ', loss)
        return loss

if __name__ == "__main__":

    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('data.csv')
    
    KM = KMeans().fit(Train_data)
    KM.predict(Validation_data)