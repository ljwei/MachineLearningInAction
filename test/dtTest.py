# _*_ coding: utf-8 _*_
__author__ = 'jwli'

from sklearn.neighbors import KNeighborsClassifier

from algorithms.classifier.decision_tree import DT
from dataModel.createDataSet import DataModel
from preprocess.utils import *
from sklearn import tree

class dtTest:

    def test1(self):
        dataModel = DataModel('dt_test')

        myTree = DT()
        myTree.fit(dataModel.trainDataSet, dataModel.trainLabels)
        print myTree.predict([1, 0])
        print myTree.predict([1, 1])
        print myTree.tree
        print myTree.labels


if "__main__" == __name__:
    test = dtTest()
    test.test1()