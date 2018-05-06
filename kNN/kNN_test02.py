# encoding: utf-8
"""
@author:max bay 
@version: python 3.6
@time: 2018/5/6 0:21
"""

import numpy as np
import operator
from os import listdir


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


def img_to_vector(filename):
    """
    将图像转换为向量：该函数创建1×1024的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    # 创建1×1024的NumPy数组
    return_vect = np.zeros((1, 1024))
    # 打开给定的文件名
    fr = open(filename)
    # 循环读出文件的前32行
    for i in range(32):
        line_str = fr.readline()
        # 将每行的头32个字符值存储在NumPy数组
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    # 返回数组
    return return_vect


def handwriting_class_test():
    """
    测试手写数字识别系统的kNN分类器
    """
    # 创建测试集标签
    hw_labels = []
    # 加载训练数据
    training_file_list = listdir('trainingDigits')
    # 获取文件夹下文件的个数
    m = len(training_file_list)
    # 创建一个m行1024列的训练矩阵，该矩阵的每行数据存储一个图像
    training_mat = np.zeros((m, 1024))
    # 从文件名中解析出分类数字，如文件9_45.txt的分类是9，它是数字9的第45个实例
    for i in range(m):
        # 获取文件名
        file_name_str = training_file_list[i]
        # 去掉 .txt
        file_str = file_name_str.split('.')[0]
        # 获取分类数字
        class_num_str = int(file_str.split('_')[0])
        # 将获取到的分类数字添加到标签向量中
        hw_labels.append(class_num_str)
        # 将每一个文件的1x1024数据存储到训练矩阵中
        training_mat[i, :] = img_to_vector('trainingDigits/%s' % file_name_str)
    # 加载测试数据
    test_file_list = listdir('testDigits')
    # 错误计数
    error_count = 0.0
    # 测试数据的个数
    m_test = len(test_file_list)
    # 从测试数据文件名中解析出分类数字
    for i in range(m_test):
        # 获取文件名
        file_name_str = test_file_list[i]
        # 去掉 .txt
        file_str = file_name_str.split('.')[0]
        # 获取分类数字
        class_num_str = int(file_str.split('_')[0])
        # 获取测试集的1x1024向量,用于训练
        vector_under_test = img_to_vector('testDigits/%s' % file_name_str)
        # 返回分类结果
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str))
        if (classifier_result != class_num_str): error_count += 1.0
    # 输出错误个数
    print("\nthe total number of errors is: %d" % error_count)
    # 输出错误率
    print("\nthe total error rate is: %f" % (error_count / float(m_test)))


if __name__ == '__main__':
    handwriting_class_test()
