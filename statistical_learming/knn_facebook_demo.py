# coding=utf-8


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class knn_demo:

    def read_data(self):
        """
        读取数据集，并进行数据处理预处理
        :return:
        """

        data = pd.read_csv("./data/knn/train.csv")

        # 把签到次数少于n的目标位置删除
        place_count = data.groupby('place_id').count()
        tf = place_count[place_count.row_id > 3].reset_index()
        data = data[data['place_id'].isin(tf.place_id)]

        # 去掉row_id
        data = data.drop(["row_id"], axis=1)

        # 处理时间数据
        date_time = pd.to_datetime(data["time"], unit='s')
        date_time = pd.DatetimeIndex(date_time)
        # 往训练集中添加特征
        data.loc[:, "day"] = date_time.day
        data.loc[:, 'hour'] = date_time.hour
        data.loc[:, 'weekday'] = date_time.weekday

        x = data[["x", "y", "accuracy"]]
        y = data["place_id"]

        # 进行数据的分割训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        # 对测试集和训练集的特征值进行标准化
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.fit_transform(x_test)

        return x_train, x_test, y_train, y_test


    def knn_classify(self, x_train, x_test, y_train, y_test):

        # 构造一些参数的值进行搜索
        params = {"n_neighbors": [3, 5]}

        knn = KNeighborsClassifier()
        gc = GridSearchCV(knn, param_grid=params, cv=3)

        gc.fit(x_train, y_train)

        y_predict = gc.predict(x_test)
        print("预测的目标签到位置为：", y_predict)
        print("在测试集上的准确率：", gc.score(x_test, y_test))
        print("在交叉验证中最好的结果:", gc.best_score_)
        print("选择最好的模型是：", gc.best_estimator_)
        print("每个超参数每次交叉验证的结果：", gc.cv_results_)


if __name__ == "__main__":
    knn = knn_demo()
    x_train, x_test, y_train, y_test = knn.read_data()
    knn.knn_classify(x_train, x_test, y_train, y_test)

