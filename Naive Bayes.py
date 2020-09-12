import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter, defaultdict

class DimensionValueError(ValueError):
    pass

class TypeError(ValueError):
    pass

class IterError(ValueError):
    pass

class DataProcess:
    # 读入数据
    def loadArffData(self, path):
        data = arff.loadarff(path) 
        df = pd.DataFrame(data[0])
        df = pd.DataFrame(df.values, columns=['sepal length','sepal width', 'petal length', 'petal width', 'class'])
        return df

    def Predata(self, path):
        df = self.loadArffData(path)

        # labels = df.values[:, -1]
        title_mapping = {b"Iris-setosa": 1, b"Iris-versicolor": 2, b"Iris-virginica": 3}#将标签对应数值
        df['class'] = df['class'].map(title_mapping)#处理数据
        df['class'] = df['class'].fillna(0)##将其余标签填充为0值

        data = df.values[:, 0:df.values.shape[-1] - 1]
        labels = df.values[:, -1]

        return data, labels

    #划分数据集
    def Split_Data(self, path, t_s=0.2): 
        data, labels = self.Predata(path)
        #为w和b和统一表示，加一维全为1的值
        # data = np.hstack((np.ones((data.shape[0], 1)), data))
        return train_test_split(data, labels, test_size = t_s, random_state = 0)

class Bayes:
    def __init__(self):
        pass

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.Train()
        return self

    def CalNormalDistribution(self, x, mean, std): 
        res = 1 / np.sqrt(2 * np.pi) / std * np.exp(-1 / 2 * (x - mean) ** 2 / std ** 2)
        return res

    def Train(self):
        unique_cls = np.unique(self.labels)
        #计算类先验概率P(yi)
        C = Counter(self.labels)
        C = sorted(C.items(), key = lambda x:(x[0], x[1]))
        Py = [cnt[1] / self.labels.shape[0] for cnt in C]
        data = []
        for i in range(unique_cls.shape[0]):
            y_ind = self.labels == C[i][0]
            data.append(self.data[y_ind])

        
        #计算条件概率P(xi|yi) 
        #假设该概率分布服从正态分布N(u,sigma^2)
        #Pxi_yi保存参数均值和标准差

        Pxi_yi = defaultdict(tuple)
        for y_i in range(unique_cls.shape[0]):
            u = np.mean(data[y_i], axis = 0)
            sigma = np.std(data[y_i], axis = 0)
            for x_i in range(self.data.shape[1]):
                Pxi_yi[(x_i, y_i)] = (u[x_i], sigma[x_i])

        self.Py = Py
        self.Likelihood = Pxi_yi
        return self

    def predict(self, data, labels):
        ycls = np.unique(self.labels)
        cnt = 0
        for i, x in enumerate(data):
            p = np.ones_like(ycls, dtype = np.float64)
            for y_i in range(ycls.shape[0]):
                p[y_i] = self.Py[y_i]
                for x_i in range(data.shape[1]):
                    mean = self.Likelihood[(x_i, y_i)][0]
                    std = self.Likelihood[(x_i, y_i)][1]
                    p[y_i] *= self.CalNormalDistribution(x[x_i], mean, std)
            print('predict: ', p, 'True label == ', labels[i] - 1)
            if np.argmax(p) == labels[i] - 1:
                cnt += 1

        print('accuracy == ', cnt / data.shape[0])
    

if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('iris.arff')

    Train_data = np.array(Train_data, dtype=np.float64)
    Validation_data = np.array(Validation_data, dtype=np.float64)
    Train_labels = np.array(Train_labels, dtype=np.int64)
    Validation_labels = np.array(Validation_labels, dtype=np.int64)

    clf = Bayes().fit(Train_data, Train_labels).predict(Validation_data, Validation_labels)
    