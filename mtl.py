import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np

#随机生成两个dataframe
d1 = pd.DataFrame(columns=['x', 'y'])
d1['x'] = np.random.normal(0, 1, 100)
d1['y'] = np.random.normal(0, 1, 100)
d2 = pd.DataFrame(columns=['x', 'y'])
d2['x'] = np.random.normal(2, 1, 100)
d2['y'] = np.random.normal(2, 1, 100)
print(d1.values)
print(d2.values)

#分别画出scatter图，但是设置不同的颜色
plt.scatter(d1['x'], d1['y'], color='blue', label='d1 points')
plt.scatter(d2['x'], d2['y'], color='green', label='d2 points')

#设置图例
plt.legend(loc=(1, 0))

#显示图片
plt.show()


# positive_index = np.nonzero( testLabelArr == 1 )
# testDateArr_positive = testDateArr[positive_index]
# negative_index = np.nonzero( testLabelArr == 0 )
# testDateArr_negative = testDateArr[negative_index]
# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.scatter(testDateArr_positive[:, 1], testDateArr_positive[:, 2], c='red')
# ax.scatter(testDateArr_negative[:, 1], testDateArr_negative[:, 2], c='black')
# ax.plot(np.arange(0,10),(-np.arange(0,10)*weight_vector[1]-weight_vector[0])/weight_vector[2])

# plt.xlabel('x', fontsize=10)
# plt.ylabel('y', fontsize=10)
# plt.show()