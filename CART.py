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
    def Split_Data(self, path, t_s=0.1): 
        data, labels = self.Predata(path)
        #为w和b和统一表示，加一维全为1的值
        # data = np.hstack((np.ones((data.shape[0], 1)), data))
        return train_test_split(data, labels, test_size = t_s, random_state = 0)

class node:
    def __init__(self, feature = -1, value = None, RChild = None, LChild = None, res = None):
        #第几个feature
        self.feature = feature 
        #feature对应的值
        self.value = value
        #右子树
        self.RChild = RChild
        #左子树
        self.LChild = LChild
        #叶节点标记
        self.res = res

class CART:
    def __init__(self):
        super().__init__()
        self.tree = None

    def fit(self, data, labels, min_sample = 1, epsilon = 1e-3):
        self.data = data
        self.labels = labels
        self.min_sample = min_sample
        self.epsilon = epsilon
        self.Train()
        return self

    #基尼指数
    def CalGini(self, labels):
        c = Counter(labels)
        return 1 - sum([(val / labels.shape[0]) ** 2 for val in c.values()])

    def CalSplitSetGini(self, Set1, Set2):
        num = Set1.shape[0] + Set2.shape[0]
        return Set1.shape[0] / num * self.CalGini(Set1) + Set2.shape[0] / num * self.CalGini(Set2)
    
    def GetBestSplit(self, split_set, data, labels):
        init_gini = self.CalGini(labels)
        split_ind = defaultdict(list)
        for split in split_set:
            for ind, sample in enumerate(data):
                if sample[split[0]] == split[1]:
                    #记住该分支的所有下标
                    split_ind[split].append(ind)
        #基尼指数最大为1
        Min_gini = 1
        best_split = None
        best_set = None
        for split, data_ind in split_ind.items():
            set1 = labels[data_ind]
            set2_ind = list(set(range(labels.shape[0])) - set(data_ind))
            set2 = labels[set2_ind]
            if set1.shape[0] == 0 or set2.shape[0] == 0:
                continue
            cur_gini = self.CalSplitSetGini(set1, set2)
            if cur_gini < Min_gini:
                Min_gini = cur_gini
                best_split = split
                best_set = (data_ind, set2_ind)
        #未超过阈值则放弃此次划分
        if abs(init_gini - Min_gini) < self.epsilon:
            best_split = None

        return best_split, best_set

    def Train(self):
        #保存划分的候选feature以及对应的val
        split_set = []
        for feature in range(self.data.shape[1]):
            uniqueVal = np.unique(self.data[:, feature])
            if uniqueVal.shape[0] < 2:
                continue
            elif uniqueVal.shape[0] == 2:
                split_set.append((feature, uniqueVal[0]))
            else :
                for val in uniqueVal:
                    split_set.append((feature, val))
        self.tree = self.BuildTree(split_set, self.data, self.labels)

    #递归建树
    def BuildTree(self, split_set, data, labels):
        if labels.shape[0] < self.min_sample:
            return node(res = Counter(labels).most_common(1)[0][0])
        best_split, best_set = self.GetBestSplit(split_set, data, labels)
        if best_split is None:
            return node(res = Counter(labels).most_common(1)[0][0])
       
        split_set.remove(best_split)
        LChild = self.BuildTree(split_set, data[best_set[0]], labels[best_set[0]])
        RChild = self.BuildTree(split_set, data[best_set[1]], labels[best_set[1]])
        return node(feature = best_split[0], value = best_split[1], RChild = RChild, LChild = LChild)

    def predict(self, data, labels):
        cnt = 0
        for x, y in zip(data, labels):
            if self.predict_x(x) == y:
                cnt += 1
        print('accuracy == ', cnt / data.shape[0])
        return self

    def predict_x(self, x):
        def Traverse(root, x):
            if root.res is not None:
                return root.res
            if x[root.feature] == root.value:
                return Traverse(root.LChild, x)
            return Traverse(root.RChild, x)
        return Traverse(self.tree, x)

    def Tree_Visual(self):
        pass

if __name__ == "__main__":
    Train_data, Validation_data, Train_labels, Validation_labels = DataProcess().Split_Data('iris.arff')

    Train_data = np.array(Train_data, dtype=np.float64)
    Validation_data = np.array(Validation_data, dtype=np.float64)
    Train_labels = np.array(Train_labels, dtype=np.int64)
    Validation_labels = np.array(Validation_labels, dtype=np.int64)
    
    clf = CART().fit(Train_data, Train_labels)
    print('In Training Data:')
    clf.predict(Train_data, Train_labels)
    print('In Validation Data:')
    clf.predict(Validation_data, Validation_labels)
