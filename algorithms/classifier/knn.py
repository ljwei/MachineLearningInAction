# _*_ coding: utf-8 _*_
__author__ = 'jwli'

import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, dataSet, labels):
        self.dataSet = dataSet
        self.labels = labels

    def predict(self, inX):
        dataSetSize = self.dataSet.shape[0]
        diffMat = np.tile(inX, (dataSetSize,1)) - self.dataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort()
        classCount={}
        for i in range(self.k):
            voteIlabel = self.labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.iteritems(), lambda a, b: cmp(a[1], b[1]), reverse=True)
        return sortedClassCount[0][0]

    def predictAll(self, inputs):
        result = []
        for input in inputs:
            result.append(self.predict(input))
        return result
