# encoding: utf-8
"""
@author:max bay 
@version: python 3.6
@time: 2018/5/5 0:25
"""

import numpy as np
import operator


def create_data_set():
    """
    函数作用：创建数据集和标签
    set_group:数据集
    set_labels:数据集对应的标签
    """
    set_group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    set_labels = ['A', 'A', 'B', 'B']
    return set_group, set_labels


def classify0(input_data, data_set, labels_set, k):
    """
    函数作用：使用k-近邻算法将每组数据划分到某个类中
    :param input_data:用于分类的输入数据(测试集)
    :param data_set:输入的训练样本集
    :param labels_set:训练样本标签
    :param k:用于选择最近邻居的数目，即kNN算法参数,选择距离最小的k个点
    :return:返回分类结果
    """
    # data_set.shape[0]返回训练样本集的行数
    data_set_size = data_set.shape[0]
    # 在列方向上重复input_data，1次，行方向上重复input_data，data_set_size次
    diff_mat = np.tile(input_data, (data_set_size, 1)) - data_set
    # diff_mat：输入样本与每个训练样本的差值,然后对其每个x和y的差值进行平方运算
    sq_diff_mat = diff_mat ** 2
    # 按行进行累加，axis=1表示按行。
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方运算，求出距离
    distances = sq_distances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sorted_dist_indices = distances.argsort()
    # 定一个字典:统计类别次数
    class_count = {}

    for i in range(k):
        # 取出前k个元素的类别
        vote_index_label = labels_set[sorted_dist_indices[i]]
        # 统计类别次数
        class_count[vote_index_label] = class_count.get(vote_index_label, 0) + 1
        # 把分类结果进行降序排序，然后返回得票数最多的分类结果
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]


if __name__ == '__main__':
    # 创建数据集
    group, labels = create_data_set()
    # 测试集
    test = [3, 0.3]
    # kNN算法进行分类
    test_class = classify0(test, group, labels, 3)
    # 显示分类结果
    print(test_class)
