# _*_ coding: utf-8 _*_
__author__ = 'jwli'

from numpy import *
from os import listdir

class DataModel:
    def __init__(self, name):
        self.trainDataSet = None
        self.trainLabels = None
        self.testDataSet = None
        self.testLabels = None
        self.features = None
        if name == 'knn_test':
            self.createKNNTestDataSet()
        elif name == 'dating':
            self.createDatingDataSet()
        elif name == 'digits':
            self.createDigitsDataSet()
        elif name == 'dt_test':
            self.createDTTestDataSet()
        elif name == 'nb_test':
            self.createNBDataSet()

    def createNBDataSet(self):
        self.trainDataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        self.trainLabels = [0,1,0,1,0,1]    #1 is abusive, 0 not

    def createDTTestDataSet(self):
        self.trainDataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
        self.trainLabels = ['no surfacing', 'flippers']
        #change to discrete values

    def createKNNTestDataSet(self):
        self.trainDataSet = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        self.trainLabels = ['A','A','B','B']

    def createDatingDataSet(self):
        filename = '../data/dating/datingTestSet2.txt'
        fr = open(filename)
        numberOfLines = len(fr.readlines())         #get the number of lines in the file
        self.trainDataSet = zeros((numberOfLines,3))        #prepare matrix to return
        self.trainLabels = []                       #prepare labels return
        fr = open(filename)
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            self.trainDataSet[index,:] = listFromLine[0:3]
            self.trainLabels.append(int(listFromLine[-1]))
            index += 1

    def img2vector(self, filename):
        returnVect = zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect

    def createDigitsDataSet(self):
        trainFile = '../data/digits/trainingDigits/'
        testFile = '../data/digits/testDigits/'

        self.trainLabels = []
        trainingFileList = listdir(trainFile)           #load the training set
        m = len(trainingFileList)
        self.trainDataSet = zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]     #take off .txt
            classNumStr = int(fileStr.split('_')[0])
            self.trainLabels.append(classNumStr)
            self.trainDataSet[i, :] = self.img2vector(trainFile + fileNameStr)

        testFileList = listdir(testFile)        #iterate through the test set
        mTest = len(testFileList)
        self.testLabels = []
        self.testDataSet  = zeros((mTest, 1024))
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]     #take off .txt
            classNumStr = int(fileStr.split('_')[0])
            self.testLabels.append(classNumStr)
            self.testDataSet[i, :] = self.img2vector(testFile + fileNameStr)
        return self.trainDataSet, self.trainLabels, self.testDataSet, self.testLabels
