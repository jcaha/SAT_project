import os
import random

datasetPath = '../data/bigclonebenchdata/'
simMatPath = '../data/similarity.txt'
testListPath = '../data/testFuncList.txt'
valListPath = '../data/valFuncList.txt'
trainListPath = '../data/trainFuncList.txt'
testPairPath = '../data/testPair.txt'
valPairPath = '../data/valPair.txt'
trainPairPath = '../data/trainPair.txt'


def writeList(contentList, path):
    with open(path, 'w') as f:
        for item in contentList:
            f.write(item)
            f.write('\t')


def readList(path):
    with open(path, 'r') as f:
        content = f.readline()
        contentList = content.strip('\t').split('\t')
    return contentList


def isTest(ind):
    return ind < testNum


def isVal(ind):
    return ind >= testNum and ind < testNum + valNum


def isTrain(ind):
    return ind >= testNum + valNum and ind < testNum + valNum + trainNum


# set args
reGenSplits = True
random.seed(996)
testNum = 500
valNum = 500
trainNum = 8000

if __name__ == "__main__":
    if not os.path.exists(trainListPath) and reGenSplits:
        funcList = []
        funcDict = {}
        for ind, func in enumerate(os.listdir(datasetPath)):
            funcList.append(func)
            funcDict[func] = ind
        random.shuffle(funcList)
        testList = funcList[:testNum]
        valList = funcList[testNum:testNum+valNum]
        trainList = funcList[testNum+valNum: testNum+valNum+trainNum]
        funcList = testList + valList + trainList 
        writeList(testList, testListPath)
        writeList(valList, valListPath)
        writeList(trainList, trainListPath)
    elif not os.path.exists(trainListPath):
        assert 0
    else:
        funcDict = {}
        for ind, func in enumerate(os.listdir(datasetPath)):
            funcDict[func] = ind
        testList = readList(testListPath)
        valList = readList(valListPath)
        trainList = readList(trainListPath)
        funcList = testList + valList + trainList

    indexDict = {}
    for ind, func in enumerate(funcList):   # map oldIndex: newIndex
        indexDict[funcDict[func]] = ind

    testPairFile = open(testPairPath, 'w')
    valPairFile = open(valPairPath, 'w')
    trainPairFile = open(trainPairPath, 'w')

    with open(simMatPath, 'r') as fmat:
        line = fmat.readline()
        firstInd = 0
        while line:
            line = line.strip('\n').split(' ')
            if len(line) < 1 or firstInd not in indexDict.keys():
                line = fmat.readline()
                firstInd += 1
                continue
            newFirstInd = indexDict[firstInd]
            firstFunc = funcList[newFirstInd]
            for secondInd, label in enumerate(line[firstInd:]):
                if secondInd not in indexDict.keys():
                    continue
                newSecondInd = indexDict[secondInd]
                secondFunc = funcList[newSecondInd]
                if label[0] == '0' or label[0] == '-':
                    label = '-1'
                else:
                    label = '1'
                pair = firstFunc + '\t' + secondFunc + '\t' + label + '\n'
                if isTest(newFirstInd) and isTest(newSecondInd):
                    testPairFile.write(pair)
                elif isVal(newFirstInd) and isVal(newSecondInd):
                    valPairFile.write(pair)
                elif isTrain(newFirstInd) and isTrain(newSecondInd):
                    trainPairFile.write(pair)

            line = fmat.readline()
            firstInd += 1

    testPairFile.close()
    valPairFile.close()
    trainPairFile.close()

