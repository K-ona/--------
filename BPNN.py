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

class BPNN:
    def __init__(self, input, List):
        super().__init__()
        self.w = []
        self.w.append(np.random.rand(List[0], input + 1))
        for i in range(len(List))[1:]:
            self.w.append(np.random.rand(List[i], List[i - 1] + 1))

    def sigmoid(self, x):
        res = 1 / (1 + np.exp(-x))
        return res

    def relu(self, x):
        res = max(0, x)
        return res

    def getw(self):
        return self.w

    def fit(self, data, labels, learning_rate):
        self.learning_rate = learning_rate
        self.data = data
        self.labels = labels
        

if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('iris.arff')
    # Train_data = Train_data.T
    # Validation_data = Validation_data.T

    Train_data = np.array(Train_data, dtype=np.float64)
    Validation_data = np.array(Validation_data, dtype=np.float64)
    Train_labels = np.array(Train_labels, dtype=np.int64)
    Validation_labels = np.array(Validation_labels, dtype=np.int64)

    BPNN = BPNN(10, [10, 10, 10]).getw()
    print(BPNN)
    

   
    
