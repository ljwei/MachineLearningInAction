# _*_ coding: utf-8 _*_
__author__ = 'jwli'

from sklearn.neighbors import KNeighborsClassifier

from algorithms.classifier.knn import KNN
from dataModel.createDataSet import DataModel
from preprocess.utils import *
class knnTest:

    def knnTest(self):
        dataModel = DataModel('knn_test')

        myknn = KNN(3)
        myknn.fit(dataModel.trainDataSet, dataModel.trainLabels)
        print "原生分类器,[0,1]的分类结果为", myknn.predict([0.8, 1])

        ## scikit learn代码
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(dataModel.trainDataSet, dataModel.trainLabels)
        print "SciKit的KNN分类器,[0,1]的分类结果为", neigh.predict([[0.8, 1]])

    def knnTest2(self):
        hoRatio = 0.20      #hold out 20%
        dataModel = DataModel('dating')
        myknn = KNN(3)
        normMat = autoNorm(dataModel.trainDataSet)
        m = normMat.shape[0]
        numTestVecs = int(m * hoRatio)
        myknn.fit(normMat[:numTestVecs, :], dataModel.trainLabels[:numTestVecs])
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = myknn.predict(normMat[i, :])
            if (classifierResult != dataModel.trainLabels[i]):
                print "分类结果为: %d, 实际结果为: %d" % (classifierResult, dataModel.trainLabels[i])
                errorCount += 1.0
        print "总错误率为: %f" % (errorCount / float(numTestVecs))
        print errorCount, "个错误"

        errorCount = 0.0
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(normMat[:numTestVecs, :], dataModel.trainLabels[:numTestVecs])
        for i in range(numTestVecs):
            classifierResult = neigh.predict(normMat[i, :])
            if (classifierResult != dataModel.trainLabels[i]):
                print "分类结果为: %d, 实际结果为: %d" % (classifierResult, dataModel.trainLabels[i])
                errorCount += 1.0
        print "SCIKIT分类总错误率为: %f" % (errorCount / float(numTestVecs))
        print errorCount

    def knnTest3(self):
        dataModel = DataModel('digits')
        myknn = KNN(3)
        myknn.fit(dataModel.trainDataSet, dataModel.trainLabels)
        errorCount = 0.0
        testNum = dataModel.testDataSet.shape[0]
        for i in range(testNum):
            classifierResult = myknn.predict(dataModel.testDataSet[i, :])
            if (classifierResult != dataModel.testLabels[i]):
                print "分类结果为: %d, 实际结果为: %d" % (classifierResult, dataModel.testLabels[i])
                errorCount += 1.0
        print "总错误率为: %f" % (errorCount / float(testNum))
        print errorCount, "个错误"

        errorCount = 0.0
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(dataModel.trainDataSet, dataModel.trainLabels)
        for i in range(testNum):
            classifierResult = neigh.predict(dataModel.testDataSet[i, :])
            if (classifierResult != dataModel.testLabels[i]):
                print "分类结果为: %d, 实际结果为: %d" % (classifierResult, dataModel.testLabels[i])
                errorCount += 1.0
        print "SCIKIT分类总错误率为: %f" % (errorCount / float(testNum))
        print errorCount

if "__main__" == __name__:
    test = knnTest()
    test.knnTest()
    test.knnTest2()
    test.knnTest3()