import os
import pickle
import numpy as np
from treeBuilder import getAstNodeList, getAstTree, getChildrenList


def getWordEmbedding(word, charEmb):
    wordEmb = np.zeros(len(charEmb))
    for ind, char in enumerate(word):
        wordEmb += (len(word) - ind) * 1.0 / len(word) * charEmb[char]
    return wordEmb


def getCharEmbedding():
    charList = ['7', 'I', 'E', 'D', 'u', 'C', 'Y', 'W', 'y', '|', '9', '^', 'X', 't', 'a', 'o', 'Z', 'b', 'A', 'J', 'R',
                'w', '?', 'g', '3', '$', 'B', 'l', '5', 'z', 'v', 'T', '2', 'd', '<', 'e', 'M', 'c', 'S', 'm', '4', 'K',
                'O', 'f', 'i', '=', 'Q', '+', 'x', 'N', '1', 'r', 'p', 'G', 'k', '*', 'q', 'L', 'P', '.', 'n', 'j', 'V',
                'U', '6', '/', '%', '8', 'F', 's', '!', '-', '&', '>', 'h', 'H', '0', '_']
    charEmbDict = {}
    for ind, char in enumerate(charList):
        charEmb = np.zeros(len(charList),dtype=np.float32)
        charEmb[ind] = 1
        charEmbDict[char] = charEmb
    return charEmbDict

def getEmbDict(embDictPath, treeList=None):
    if os.path.isfile(embDictPath):
        embeddingDict = pickle.load(open(embDictPath,'rb'))
        return embeddingDict
    if treeList is None:
        print("please input treeList to getEmbDict")
        assert 0

    embeddingDict = {}
    wordList = []
    charEmbedding = getCharEmbedding()

    print("-------------- start generate embedding ---------------")

    for num, tree in enumerate(treeList.values()):
        wordList.extend(getAstNodeList(tree))
        wordList = list(set(wordList))
        if num%300 == 299:
            print('{} trees\' words embedded!'.format(num+1))
    for word in wordList:
        embeddingDict[word] = getWordEmbedding(word, charEmbedding)
    pickle.dump(embeddingDict, open(embDictPath,'wb'))
    return embeddingDict


def getDataList(treeDict, wordEmb):
    '''
    treeDict: treeName -> treeRoot
    wordEmb: wordStr -> embedding
    '''
    childrenListDict = {}
    treeEmb = {}
    for treeName in treeDict.keys():
        root = treeDict[treeName]
        childrenList, wordList = getChildrenList(root)
        embList = []
        for word in wordList:
            embList.append(wordEmb[word])

        childrenListDict[treeName] = childrenList
        treeEmb[treeName] = embList
    return childrenListDict, treeEmb


if __name__ == "__main__":
    datasetPath = '../data/bigclonebenchdata/'
    embPath = '../data/wordEmbedding.pkl'
    treeList = []
    print("----------------- start building tree -----------------")
    for ind, sample in enumerate(os.listdir(datasetPath)):
        path = os.path.join(datasetPath, sample)
        root, nodeNum = getAstTree(path)
        treeList.append(root)
        if ind%300 == 299:
            print('{} trees finished.'.format(ind+1))

    getEmbDict(embPath, treeList=treeList)
