# coding=utf-8

import numpy as np

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = "./data/iris.data"  # 数据文件路径

    # 路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    print(data)
    # 将数据的0到3列组成x，第4列得到y
    x, y = np.split(data, (4,), axis=1)