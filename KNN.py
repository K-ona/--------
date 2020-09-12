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
    def Split_Data(self, path, t_s=0.2):
        data, labels = self.Predata(path)
        #为w和b和统一表示，加一维全为1的值
        # data = np.hstack((np.ones((data.shape[0], 1)), data))
        return train_test_split(data, labels, test_size = t_s, random_state = 0)

class KNN:
    def __init__(self):
        pass

    def fit(self, data, labels, k = 7):
        self.data = data
        self.labels = labels
        self.k = k
        return self

    #计算x, y之间的距离
    def cal_dis(self, x, y):
        return np.linalg.norm(x - y)
    
    def dicision(self, kresult):
        #从训练集中定义类别
        sorts = np.unique(self.labels)
        
        res_dict = { x:0 for x in sorts.tolist()}
        for res in kresult:
            res_dict[self.labels[int(res)]] += 1
        
        sorted_dict = sorted(zip(res_dict.values(), res_dict.keys()))
        return sorted_dict[-1][1]
        
    def predict(self, Valdata, Vallabels):
        labels = np.zeros((Valdata.shape[0]))
        for i in range(Valdata.shape[0]):
            dis = np.array([ [self.cal_dis(Valdata[i], self.data[j]), j] for j in range(self.data.shape[0])])
            tmp = pd.DataFrame(dis)
            sorted = tmp.sort_values(by = 0).values
            labels[i] = self.dicision(sorted[:self.k, 1])
        self.error(Valdata, labels, Vallabels)
        self.visual(Valdata, Vallabels)
        return self

    def datasplit(self, data, labels):
        tmp = np.hstack((labels.reshape((-1, 1)), data))
        df = pd.DataFrame(tmp, columns=['lable', 'x', 'y'])

        data1 = df.loc[df.lable == 1.0]
        data2 = df.loc[df.lable == 0]    
        return data1, data2

    def visual(self, data, labels):

        datap, datan = self.datasplit(self.data, self.labels)
        plt.scatter(datap['x'], datap['y'], color='blue', label='negative')
        plt.scatter(datan['x'], datan['y'], color='green', label='positive')
        plt.legend()
        plt.show()

        datap, datan = self.datasplit(data, labels)
        plt.scatter(datap['x'], datap['y'], color='blue', label='negative')
        plt.scatter(datan['x'], datan['y'], color='green', label='positive')
        plt.legend()
        plt.show()
        pass

    def error(self, data, result, labels):
        error_rate = labels[labels != result].shape[0] / labels.shape[0]
        print('error_rate = ', error_rate, 'with k = ', self.k)
        pass

if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('data.csv')

    Knn = KNN().fit(Train_data, Train_labels).predict(Validation_data, Validation_labels)
