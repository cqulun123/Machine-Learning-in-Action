# encoding: utf-8
"""
@author:max bay 
@version: python 3.6
@time: 2018/5/5 22:04
"""

import numpy as np
import operator


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


def file_to_matrix(filename):
    """
     函数作用：从文件中读入训练数据，并存储为矩阵
    :param filename:文件名字符串
    :return:训练样本矩阵和类标签向量
    """
    # 打开文件
    fr = open(filename)
    # 读取文件内容
    array_lines = fr.readlines()
    # 得到文件行数
    number_of_lines = len(array_lines)
    # 返回解析后的数据
    return_mat = np.zeros((number_of_lines, 3))
    # 定义类标签向量
    class_label_vector = []
    # 行索引值
    index = 0
    for line in array_lines:
        # 去掉 回车符号
        line = line.strip()
        # 用\t分割每行数据
        list_from_line = line.split('\t')
        # 选取前3个元素，将它们存储到特征矩阵中
        return_mat[index, :] = list_from_line[0:3]
        # 把该样本对应的标签放至标签向量，顺序与样本集对应。
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    该函数可以自动将数字特征值转化为0到1的区间,即归一化训练数据
    """
    # 获取数据集中每列的最小数值
    min_vals = data_set.min(0)
    # 获取数据集中每列的最大数值
    max_vals = data_set.max(0)
    # 最大值与最小的差值
    ranges = max_vals - min_vals
    # 创建一个全0矩阵，用于存放归一化后的数据
    norm_data_set = np.zeros(np.shape(data_set))
    # 返回data_set的行数
    m = data_set.shape[0]
    # 原始数据值减去最小值
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    # 除以最大和最小值的差值,得到归一化数据
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    # 返回归一化数据，最大值与最小的差值，每列的最小数值
    return norm_data_set, ranges, min_vals


def dating_class_test():
    # 将数据集中90%用于训练,10%的数据留作测试用
    ho_ratio = 0.10
    # 将返回的特征矩阵和分类向量分别存储到dating_data_mat和dating_labels中
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet2.txt')
    # 返回归一化数据，最大值与最小的差值，每列的最小数值
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 获得norm_mat的行数
    m = norm_mat.shape[0]
    # 10%的测试数据的个数
    num_test_vecs = int(m * ho_ratio)
    # 分类错误计数
    error_count = 0.0
    for i in range(num_test_vecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]): error_count += 1.0
    # 错误率
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def classify_person():
    # 类标签列表
    result_list = ['not at all', 'in small doses', 'in large doses']
    # 用户输入不同特征值
    precent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumedper year?"))
    # 打开的文件名并处理数据
    dating_data_mat, dating_labels = file_to_matrix("datingTestSet2.txt")
    # 归一化训练集
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 创建测试集数组
    in_arr = np.array([precent_tats, ff_miles, ice_cream])
    # 归一化测试集
    norm_in_arr = (in_arr - min_vals) / ranges
    # 返回分类结果
    classifier_result = classify0(norm_in_arr, norm_mat, dating_labels, 3)
    # 输出结果
    print("You will probably like this person: ", result_list[classifier_result - 1])


if __name__ == '__main__':
        classify_person()
