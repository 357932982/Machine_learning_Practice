# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd


# 获取数据
def get_data():
    # 读取csv文件
    data = pd.read_csv("./data/perceptron.csv")
    data = data.drop(["Id"], axis=1)

    # 返回特征集和结果集
    return data[["X1", "X2"]].values, data["Y"]


# 感知机分类
def perceptron_classify(x, y, eta=1, omiga=np.array([0, 0]), b=0, step=100):
    # 全部分类完毕或者迭代step步则结束
    i = 0
    while i < len(x) and step > 0:
        if y[i] * (np.dot(omiga, x[i]) + b) <= 0:
            omiga = omiga + eta * y[i] * x[i]
            b = b + eta * y[i]
            print("x{}被误分类，更新后omiga={}，b={}".format(i + 1, omiga, b))
            i = 0
        else:
            i += 1
        step -= 1

    print(omiga, b)
    return omiga, b


# 绘制图形
def plot(x, y, omiga, b):
    fig, ax = plt.subplots()
    x1_point = x[:, 0]  # 取第一列
    x2_point = x[:, 1]  # 取第二列

    for i in range(len(y)):
        if y[i] > 0:
            ax.scatter(x1_point[i], x2_point[i], color='red')
        else:
            ax.scatter(x1_point[i], x2_point[i], color='green')

    # 把x轴的刻度间隔设置为1，并存在变量里
    x_major_locator = MultipleLocator(1)
    # 把y轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(1)
    # 把x,y轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(0, 7)
    plt.ylim(0, 7)

    x1 = np.linspace(0, 10)
    x2 = -(omiga[0] * x1 + b) / omiga[1]
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(x1, x2)
    plt.show()


if __name__ == "__main__":
    x, y = get_data()
    omiga, b = perceptron_classify(x, y)
    plot(x, y, omiga, b)
