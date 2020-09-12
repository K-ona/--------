
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

class DataProcess:
    # 读入数据
    def loadArffData(self, path):
        data = arff.loadarff(path) 
        df = pd.DataFrame(data[0]) 
        return df
    #预处理
    def Predata(self, path):
        df = self.loadArffData(path)
        labels = df.values[:, -1]
        labels = labels == b'tested_positive'
        data = df.values[:, 0:df.values.shape[-1] - 1].T
        return data.T, labels.astype(np.int)
    #划分数据集
    def Split_Data(self, path):
        data, labels = self.Predata(path)
        #为w和b和统一表示，加一维全为1的值
        # data = np.hstack((np.ones((data.shape[0], 1)), data))
        return train_test_split(data, labels, test_size = 0.2, random_state = 32)

def csv_read(path):
    f = csv.reader(open(path))
    l = []
    for row in f:
        l.append(row)
    l = np.array(l)
 

if __name__ == "__main__":
    csv_read('data.csv')
    