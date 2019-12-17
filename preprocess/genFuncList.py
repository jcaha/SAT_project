import os
import random
import numpy as np

datasetPath = '../data/bigclonebenchdata/'
funcMapPath = '../data/functions.txt'
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

def writePair(contentListIndex, funcList, funcDict, simMat, pairPath):
    with open(pairPath, 'w') as f:
        funcNum = len(contentListIndex)
        for i in range(funcNum):
            for j in range(i+1, funcNum):
                x = contentListIndex[i]  # ind in funcList
                y = contentListIndex[j]
                x_name = funcList[x]     # func_name
                y_name = funcList[y]
                x_mat_ind = funcDict[x_name]    # func_mat_ind
                y_mat_ind = funcDict[y_name]
                label = simMat[x_mat_ind,y_mat_ind]
                f.write(x_name + '\t' + y_name + '\t' + str(label) + '\n')

# set args
random.seed(996)
testNum = 500
valNum = 500
trainNum = 8000

if __name__ == "__main__":
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

    funcDict = {}   # func_name: id in simMat
    with open(funcMapPath, 'r') as f:
        ind = 0
        while True:
            line = f.readline()
            line = line.split('\t')
            if line[0] == 'FUNCTION_ID:':
                func = line[1].rstrip('\n').rstrip('\r') + '.txt'
                funcDict[func] = ind
                ind += 1
                if ind == 9134:
                    break

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
                if like[0] =='-' or like[0] =='0':
                    like = -1
                elif like[0] == '1':
                    like = 1
                simMat[indx, indy] = like
        writePair(trainListIndex, funcList, funcDict, simMat, trainPairPath)
        writePair(testListIndex, funcList, funcDict, simMat, testPairPath)
        writePair(valListIndex, funcList, funcDict, simMat, valPairPath)
