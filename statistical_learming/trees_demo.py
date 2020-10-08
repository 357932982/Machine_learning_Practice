"""
_*_ coding: utf-8 _*_
@Time : 2020/10/5 20:55
@Author : yan_ming_shi
@Version：V 0.1
@File : trees_demo.py
@desc : 决策树相关练习
"""
import math
import operator
import pickle


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_ent(data_set):
    """
    计算给定数据集的香农熵
    :param data_set: 数据集
    :return: 香农熵
    """
    num_entries = len(data_set)
    label_counts = {}
    # 为所有可能分类创建字典
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0

    # 以二为底求对数
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)

    print("信息熵为：{}".format(shannon_ent))
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    按特征划分数据集
    :param data_set: 待划分数据集
    :param axis: 划分数据集的特征（按某列划分）
    :param value: 需要返回的特征值
    :return: 
    """
    # 创建新的list对象
    retdata_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 抽取
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            retdata_set.append(reduced_feat_vec)

    print("划分后的集合为:{}".format(retdata_set))
    return retdata_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式
    :param data_set: 待划分数据集
    :return: 
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1

    # 创建唯一分类标签
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0.0

        # 计算每种划分的信息墒
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
            info_gain = base_entropy - new_entropy

            # 计算最好的增益墒
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

    return best_feature


def majority_cnt(class_list):
    """
    获取出现次数最多的分类名称
    :param class_list: 分类集合
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count


def create_tree(data_set, labels):
    """
    创建树
    :param data_set: 数据集
    :param labels: 标签列表
    :return:
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        # 停止分类直至所有类别相等
        return class_list[0]
    if len(data_set[0]) == 1:
        # 停止分割直至没有更多特征
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])

    # 得到包含所有属性的列表
    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)

    print("创建的树为：{}".format(my_tree))
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类
    :param input_tree:
    :param feat_labels:
    :param test_vec:
    :return:
    """
    class_label = None
    first_str = list(input_tree.keys()[0])
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] ==  key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(inputtree, filename):
    """
    保存训练好的决策树模型
    :param inputtree:
    :param filename:
    :return:
    """
    fw = open(filename, 'w')
    pickle.dump(inputtree, fw)
    fw.close()


def grab_tree(filename):
    """
    加载决策树模型
    :param filename:
    :return:
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    data_set, labels = create_data_set()
    # calc_shannon_ent(data_set)
    # split_data_set(data_set, 0, 1)
    # choose_best_feature_to_split(data_set)
    create_tree(data_set, labels)
