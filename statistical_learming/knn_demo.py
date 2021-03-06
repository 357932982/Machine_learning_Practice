# coding=utf-8

import numpy as np
import operator
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from os import listdir


def create_date_set():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']

    return group, label


def classify(in_x, data_set, labels, k):
    """
    简单knn分类算法示例
    :param in_x: 待预测特征数据
    :param data_set: 已确定标签数据特征
    :param labels: 标签集合
    :param k: 邻居数量
    :return:
    """
    dataset_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (dataset_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def transform_label(label, inverse=False):
    """
    标签转换
    :param label:
    :param inverse: 是否反转
    :return:
    """
    lb = preprocessing.LabelEncoder()
    lb.fit(['didntLike', 'smallDoses', 'largeDoses'])
    if inverse:
        return lb.inverse_transform(label)
    else:
        return lb.transform(label)


def get_dating_set():
    """
    读取约会数据并对特征归一化
    :return:
    """
    data = pd.read_table("./data/knn/datingTestSet.txt", names=['a', 'b', 'c', 'd'])
    x = data[['a', 'b', 'c']]
    y = transform_label(data['d'])

    return x.values, y


def auto_norm(data):
    """
    归一化处理
    :param data:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def dating_class_test():
    """
    测试分类器预测准确率
    :return:
    """
    ho_ratio = 0.50  # hold out 10%
    dating_data_mat, dating_labels = get_dating_set()  # load data setfrom file
    norm_mat = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                          dating_labels[num_test_vecs:m], 3)
        print(
            "the classifier came back with: {}, the real answer is: {}".format(classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is:{}".format(error_count / float(num_test_vecs)))
    print(error_count)


def draw_dating_map(data, label):

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    fig, ax1 = plt.subplots()

    a = np.empty(shape=[0, 3], dtype=np.ndarray)  # didntLike
    b = np.empty(shape=[0, 3], dtype=np.ndarray)  # largeDoses
    c = np.empty(shape=[0, 3], dtype=np.ndarray)  # smallDoses
    for i in range(len(label)):
        if label[i] == 0:
            x = np.array(data[i, :]).reshape(1, 3)
            print(type(x))
            a = np.append(a, np.array(data[i, :]).reshape(1, 3), axis=0)
        elif label[i] == 1:
            b = np.append(b, np.array(data[i, :]).reshape(1, 3), axis=0)
        else:
            c = np.append(c, np.array(data[i, :]).reshape(1, 3), axis=0)

    ax1.scatter(a[:, 0], a[:, 1], label='didntLike', color='black')
    ax1.scatter(b[:, 0], b[:, 1], label='largeDoses', color='green')
    ax1.scatter(c[:, 0], c[:, 1], label='smallDoses', color='blue')
    plt.xlabel("每年获取的飞行常客里程数")
    plt.ylabel("玩视频游戏所耗时间百分比")
    # 显示图例
    plt.legend()

    fig, ax2 = plt.subplots()
    ax2.scatter(a[:, 0], a[:, 2], label='didntLike', color='black')
    ax2.scatter(b[:, 0], b[:, 2], label='largeDoses', color='green')
    ax2.scatter(c[:, 0], c[:, 2], label='smallDoses', color='blue')
    plt.xlabel("每年获取的飞行常客里程数")
    plt.ylabel("玩视频游戏所耗时间百分比")
    # 显示图例
    plt.legend()

    fig, ax3 = plt.subplots()
    ax3.scatter(a[:, 1], a[:, 2], label='didntLike', color='black')
    ax3.scatter(b[:, 1], b[:, 2], label='largeDoses', color='green')
    ax3.scatter(c[:, 1], c[:, 2], label='smallDoses', color='blue')
    plt.xlabel("玩视频游戏所耗时间百分比")
    plt.ylabel("每周消费的冰淇淋公升数")

    # 显示图例
    plt.legend()
    plt.show()


def img2vector(filename):
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def hand_writing_class_test():
    hw_labels = []
    training_file_list = listdir('./data/digits/trainingDigits')  # load the training set
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('./data/digits/trainingDigits/{}'.format(file_name_str))
    test_file_list = listdir('./data/digits/testDigits')  # iterate through the test set
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('./data/digits/testDigits/{}'.format(file_name_str))
        classifier_result = classify(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\nthe total number of errors is: {}".format(error_count))
    print("\nthe total error rate is: {}".format(error_count / float(m_test)))


if __name__ == "__main__":
    # group, label = knn.create_date_set()
    # y = knn.classify([0, 0], group, label, 3)
    # print(y)

    # x, y = knn.get_dating_set()
    # draw_dating_map(x, y)
    # dating_class_test()
    hand_writing_class_test()