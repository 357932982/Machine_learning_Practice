# coding=utf-8

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    path = "./data/advertising.csv"
    """
    手写读取数据
    """
    # f = open(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = list(map(float, d.split(',')))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # print(x)
    # print(y)
    # x = np.array(x)
    # y = np.array(y)
    """
    Python自带库
    """
    # with open(path, 'rt') as file:
    #     b = csv.reader(file)
    #     row = [row for row in b]
    # x = []
    # y = []
    # for index, i in enumerate(row):
    #     if index == 0:
    #         continue
    #     x.append(i[1:-1])
    #     y.append(i[-1])
    # x = np.array(x)
    # y = np.array(y)

    """
    numpy读取
    """
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # x = p[:, [1, 2, 3]]
    # y = p[:, [-1]]
    # print(x)
    # print(y)

    """
    用pandas读取
    """
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    print(x)
    print(y)

    # 绘图1
    # plt.plot(x['TV'], y, 'ro', label='TV')
    # plt.plot(x['Radio'], y, 'g^', label='Radio')
    # plt.plot(x['Newspaper'], y, 'mv', label='Newspaper')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    # 绘图2
    plt.figure(figsize=(9, 12))
    plt.subplot(311)
    plt.plot(x['TV'], y, 'ro')
    plt.title('Tv')
    plt.grid()
    plt.subplot(312)
    plt.plot(x['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(x['Newspaper'], y, 'mv')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    print(x_train, y_train)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print(linreg.coef_)
    print(linreg.intercept_)

    y_predice = linreg.predict(x_test)
    mse = np.average((y_predice - y_test) ** 2)  # Mean squared error
    rmse = np.sqrt(mse)  # Root mean squared error
    print("平均误差：", mse, "均方误差：", rmse)
    linreg.score()

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_predice, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
