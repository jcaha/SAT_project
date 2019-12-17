import os
import random
import numpy as np

datasetPath = '../data/bigclonebenchdata/'
simMatPath = '../data/similarity.txt'
testListPath = '../data/testFuncList.txt'
valListPath = '../data/valFuncList.txt'
trainListPath = '../data/trainFuncList.txt'
testPairPath = '../data/testPair.txt'
valPairPath = '../data/valPair.txt'
trainPairPath = '../data/trainPair.txt'


def writeList(contentListIndex, funcList, path):
    with open(path, 'w') as f:
        for item in contentListIndex:
            f.write(funcList[item])
            f.write('\t')

def writePair(contentListIndex, funcList, simMat, pairPath):
    with open(pairPath, 'w') as f:
        funcNum = len(contentListIndex)
        for i in range(funcNum):
            for j in range(i+1, funcNum):
                x = contentListIndex[i]
                y = contentListIndex[j]
                x_name = funcList[x]
                y_name = funcList[y]
                label = simMat[x,y]
                f.write(x_name + '\t' + y_name + '\t' + str(label) + '\n')

# set args
random.seed(996)
testNum = 500
valNum = 500
trainNum = 8000

if __name__ == "__main__":
    # if not os.path.exists(trainListPath) and reGenSplits:
    #     funcList = []
    #     funcDict = {}
    #     for ind, func in enumerate(os.listdir(datasetPath)):
    #         funcList.append(func)
    #         funcDict[func] = ind
    #     random.shuffle(funcList)
    #     testList = funcList[:testNum]
    #     valList = funcList[testNum:testNum+valNum]
    #     trainList = funcList[testNum+valNum: testNum+valNum+trainNum]
    #     funcList = testList + valList + trainList
    #     writeList(testList, testListPath)
    #     writeList(valList, valListPath)
    #     writeList(trainList, trainListPath)
    # elif not os.path.exists(trainListPath):
    #     assert 0
    # else:
    #     funcDict = {}
    #     for ind, func in enumerate(os.listdir(datasetPath)):
    #         funcDict[func] = ind
    #     testList = readList(testListPath)
    #     valList = readList(valListPath)
    #     trainList = readList(trainListPath)
    #     funcList = testList + valList + trainList

    # indexDict = {}
    # for ind, func in enumerate(funcList):   # map oldIndex: newIndex
    #     indexDict[funcDict[func]] = ind

    # testPairFile = open(testPairPath, 'w')
    # valPairFile = open(valPairPath, 'w')
    # trainPairFile = open(trainPairPath, 'w')

    # with open(simMatPath, 'r') as fmat:
    #     line = fmat.readline()
    #     firstInd = 0
    #     while line:
    #         line = line.strip('\n').split(' ')
    #         if len(line) < 1 or firstInd not in indexDict.keys():
    #             line = fmat.readline()
    #             firstInd += 1
    #             continue
    #         newFirstInd = indexDict[firstInd]
    #         firstFunc = funcList[newFirstInd]
    #         for secondInd, label in enumerate(line[firstInd:]):
    #             if secondInd not in indexDict.keys():
    #                 continue
    #             newSecondInd = indexDict[secondInd]
    #             secondFunc = funcList[newSecondInd]
    #             if label[0] == '0' or label[0] == '-':
    #                 label = '-1'
    #             else:
    #                 label = '1'
    #             pair = firstFunc + '\t' + secondFunc + '\t' + label + '\n'
    #             if isTest(newFirstInd) and isTest(newSecondInd):
    #                 testPairFile.write(pair)
    #             elif isVal(newFirstInd) and isVal(newSecondInd):
    #                 valPairFile.write(pair)
    #             elif isTrain(newFirstInd) and isTrain(newSecondInd):
    #                 trainPairFile.write(pair)

    #         line = fmat.readline()
    #         firstInd += 1

    # testPairFile.close()
    # valPairFile.close()
    # trainPairFile.close()
    if os.path.exists(trainListPath):
        print('there has been trainList.')
        assert 0
    funcList = []
    for func in os.listdir(datasetPath):
        funcList.append(func)
    funcNum = len(funcList)

    shuffled_index = [i for i in range(funcNum)]
    random.shuffle(shuffled_index)

    trainListIndex = shuffled_index[:trainNum]
    valListIndex = shuffled_index[trainNum:trainNum+valNum]
    testListIndex = shuffled_index[trainNum+valNum: trainNum+valNum+testNum]

    writeList(trainListIndex, funcList, trainListPath)
    writeList(testListIndex, funcList, testListPath)
    writeList(valListIndex, funcList, valListPath)

    with open(simMatPath, 'r') as f:
        data = f.read()
        dataMat = data.strip('\n').split('\n')
        size = len(dataMat)
        simMat = np.zeros((size, size), dtype=np.int16)
        for indx, line in enumerate(dataMat):
            simLine = line.strip(' ').split(' ')
            for indy, like in enumerate(simLine):
                if indx == indy:
                    continue
                if like[0]=='-' or like[0]=='0':
                    like = -1
                elif like[0] == '1':
                    like = 1
                simMat[indx, indy] = like
        writePair(trainListIndex, funcList, simMat, trainPairPath)
        writePair(testListIndex, funcList, simMat, testPairPath)
        writePair(valListIndex, funcList, simMat, valPairPath)
