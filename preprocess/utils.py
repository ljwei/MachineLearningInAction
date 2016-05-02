# _*_ coding: utf-8 _*_
__author__ = 'jwli'

from numpy import *

def autoNorm(dataSet):
    """
    归一化处理函数，以防止参数之间的值差距过大
    使得矩阵所有行的所有值均在-1至1之间
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))   #element wise divide
    return normDataSet