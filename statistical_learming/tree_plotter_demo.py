"""
_*_ coding: utf-8 _*_
@Time : 2020/10/7 22:59
@Author : yan_ming_shi
@Version：V 0.1
@File : tree_plotter_demo.py
@desc : matplotlib绘制树形图
"""

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')
# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']


# 绘制带箭头的注释
def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


# 获取叶子节点的数目
def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]

    # 测试节点数据是否为字典
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


# 获取决策树的层数
def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]

    # 测试数据是否为字典
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# 在父节点中填充文本的信息
def plot_mid_text(cntrpt, parentpt, txtstring):
    xmid = (parentpt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    ymid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    create_plot.ax1.text(xmid, ymid, txtstring, va="center", ha="center", rotation=30)


# 绘制树
def plot_tree(mytree, parentpt, nodetxt):
    # 计算树的宽和高
    numleafs = get_num_leafs(mytree)
    firststr = list(mytree.keys())[0]
    cntrpt = (plot_tree.xoff + (1.0 + float(numleafs)) / 2.0 / plot_tree.totalw, plot_tree.yoff)
    plot_mid_text(cntrpt, parentpt, nodetxt)
    plot_node(firststr, cntrpt, parentpt, decision_node)
    seconddict = mytree[firststr]
    plot_tree.yoff = plot_tree.yoff - 1.0 / plot_tree.totald

    # 测试数据是否为字典
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            plot_tree(seconddict[key], cntrpt, str(key))

        else:  # 它是叶子节点打印叶子节点
            plot_tree.xoff = plot_tree.xoff + 1.0 / plot_tree.totalw
            plot_node(seconddict[key], (plot_tree.xoff, plot_tree.yoff), cntrpt, leaf_node)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntrpt, str(key))
    plot_tree.yoff = plot_tree.yoff + 1.0 / plot_tree.totald


def create_plot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plot_tree.totalw = float(get_num_leafs(intree))
    plot_tree.totald = float(get_tree_depth(intree))
    plot_tree.xoff = -0.5 / plot_tree.totalw
    plot_tree.yoff = 1.0
    plot_tree(intree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':

    trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 2: 'maybe'}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    create_plot(trees[1])
