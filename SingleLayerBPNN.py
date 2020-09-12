import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
        title_mapping = {b"Iris-setosa": np.array([1, 0, 0]), b"Iris-versicolor": np.array([0, 1, 0]), b"Iris-virginica": np.array([0, 0, 1])}#将标签对应数值
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

class BPNN:
    def __init__(self, inputdim, hiddendim, outputdim):
        super().__init__()
        self.w = 5 * np.random.rand(outputdim, hiddendim)
        self.wb = 5 * np.random.rand(outputdim)
        self.v = 5 * np.random.rand(hiddendim, inputdim)
        self.vb = 5 * np.random.rand(hiddendim)

    def sigmoid(self, x):
        res = 1 / (1 + np.exp(-x))
        return res

    def relu(self, x):
        res = max(0, x)
        return res

    def softmax(self, list):
        pass
        return list

    def update(self):
        loss = 0
        for i in range(self.data.shape[0]):
            x = self.data[i]
            true_y = self.labels[i]

            input_b = np.dot(x, self.v.T)
            b = self.sigmoid(input_b - self.vb)
            input_y = np.dot(b, self.w.T) / 100
            y = self.sigmoid(input_y - self.wb)
            
            loss += 0.5 * np.linalg.norm(y - self.labels[i])
            
            a = np.multiply(np.mat(y), np.mat(1 - y))
            g = np.multiply(a, np.mat(y - true_y))
            e = np.multiply(np.mat(np.multiply(np.mat(b), np.mat(1 - b))), np.mat(np.dot(g, self.w)))

            deltw = self.learning_rate * np.multiply(np.mat(g).T, np.mat(b))
            deltwb = -1 * self.learning_rate * g
            deltv = self.learning_rate * np.multiply(np.mat(e).T, np.mat(x))
            deltvb = -1 * self.learning_rate * e

            self.w -= deltw
            self.wb -= np.array(deltwb).reshape((self.wb.shape[0]))
            self.v -= deltv
            self.vb -= np.array(deltvb).reshape((self.vb.shape[0]))

        # loss /= self.data.shape[0]
        print('loss == ', loss)     
        pass

    def fit(self, data, labels, activation, learning_rate = 0.1, epoch = 15000):
        self.learning_rate = learning_rate
        self.data = data
        self.labels = labels
        self.activation = activation
        for i in tqdm(range(epoch)):
            self.update()
        
        return self
 
    def predict(self, data, labels):
        loss = 0
        cnt = 0
        for i in range(data.shape[0]):
            x = data[i]
            true_y = labels[i]

            input_b = np.dot(x, self.v.T)
            b = self.sigmoid(input_b - self.vb)
            input_y = np.dot(b, self.w.T)
            y = self.sigmoid(input_y - self.wb)

            ind = np.argmax(y)
            cnt += true_y[ind]

            loss += 0.5 * np.linalg.norm(y - labels[i])
        print('accuracy == ', cnt / data.shape[0])

if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('iris.arff')

    Train_data = np.array(Train_data, dtype=np.float64)
    Train_data /= np.sum(Train_data, axis = 1).reshape((Train_data.shape[0], 1))
    Validation_data = np.array(Validation_data, dtype=np.float64)
    Validation_data /= np.sum(Validation_data, axis = 1).reshape(Validation_data.shape[0], 1)
   
    BPNN = BPNN(4, 10, 3).fit(Train_data,
                             Train_labels, activation='sigmoid').predict(Validation_data, Validation_labels)
