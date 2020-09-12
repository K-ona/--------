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
    def Split_Data(self, path, t_s=0.5): 
        data, labels = self.Predata(path)
        #为w和b和统一表示，加一维全为1的值
        # data = np.hstack((np.ones((data.shape[0], 1)), data))
        return train_test_split(data, labels, test_size = t_s, random_state = 0)

class SoftMax:
    def __init__(self):
        super().__init__()
        pass

    def fit(self, data, labels, epoch = 50, learning_rate = 0.1, threshold = 0.5):    
        self.data = data
        self.data = np.insert(self.data, 0, 1, axis=1)

        self.labels = labels
        self.sorts = np.unique(self.labels)

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.w = np.random.rand(self.sorts.shape[0], self.data.shape[1])
        
        self.BGA()
        return self

    def BGA(self):
        for i in tqdm(range(self.epoch)):
            self.w += self.learning_rate * self.getGD()
            loss = self.loss()
            print('loss = ', loss)    


    def getGD(self):
        G = np.zeros((self.sorts.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[0]):
            G[self.labels[i] - 1][i] = 1
        
        H = self.hypothesis(self.w, self.data)
        return np.dot((G - H), self.data)

    def hypothesis(self, w, data):
        eta = np.dot(w, data.T)
        H = np.exp(eta)
        H /= np.sum(H, axis = 0)
        self.H = H
        return H 

    def predict(self, Vdata, Vlabels):
        
        Vdata = np.insert(Vdata, 0, 1, axis=1)
        predict_score = self.hypothesis(self.w, Vdata)
        predict_labels = np.argmax(predict_score, axis = 0)
        TP_TN = 0
        test_size = len(predict_labels)
        for i in range(test_size):
            if predict_labels[i] + 1 == Vlabels[i]:
                TP_TN += 1
        print("准确率：", TP_TN / test_size)


    def loss(self):
        #  predict, labels
        
        log = np.log(self.H)
        sum = np.sum(log, axis=0)
       
        loss = 0.0
        for i in range(self.data.shape[0]):
            loss += log[self.labels[i] - 1, i] / sum[i]
        
        return 1 * loss / self.data.shape[0]
        
    
if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('iris.arff')
    # Train_data = Train_data.T
    # Validation_data = Validation_data.T

    Train_data = np.array(Train_data, dtype=np.float64)
    Validation_data = np.array(Validation_data, dtype=np.float64)
    Train_labels = np.array(Train_labels, dtype=np.int64)
    Validation_labels = np.array(Validation_labels, dtype=np.int64)

    SoftMax = SoftMax().fit(Train_data, Train_labels, epoch = 100)

    SoftMax.predict(Validation_data, Validation_labels)

    pass