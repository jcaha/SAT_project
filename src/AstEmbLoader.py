import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from treeBuilder import getChildrenList


def alignChildren(childrenList):
    '''
    childrenList: N x nodeNum x childrenNum
    '''
    maxNum = max([len(children) for s in childrenList for children in s])
    returnList = [[c+[0]*(maxNum-len(c)) for c in s] for s in childrenList]
    # returnList = [[[np.array(1)] for c in s] for s in childrenList]
    # print(returnList[0][50])
    # print(len(returnList[0][2]))
    maxNode = max([len(s) for s in childrenList])
    returnList = [s+[[0]*maxNum]*(maxNode-len(s)) for s in returnList]
    return returnList


def alignEmbedding(embList):
    maxNode = max([len(emb) for emb in embList])
    emb_size = len(embList[0][0])
    embList = [emb + [np.zeros(emb_size)]*(maxNode-len(emb)) for emb in embList]
    return embList


class AstEmbDataset(Dataset):
    def __init__(self, pairList, treeList, treeEmb, batch_size=1):
        self.pairList = pairList
        self.treeList = treeList
        self.batch_size = batch_size
        self.treeEmb = treeEmb

    def __getitem__(self, index):
        pairs = self.pairList[index*self.batch_size: (index+1)*self.batch_size]
        roots1 = []
        roots2 = []
        embs1 = []
        embs2 = []
        labels = []
        for pair in pairs:
            root1, root2, label = pair.strip('\n').split('\t')
            label = float(label)
            childrenList1, emb1 = self.getData(root1)
            childrenList2, emb2 = self.getData(root2)
            roots1.append(childrenList1)
            roots2.append(childrenList2)
            embs1.append(emb1)
            embs2.append(emb2)
            labels.append(label)
        roots = alignChildren(roots1 + roots2)
        roots = np.array(roots)
        embs = alignEmbedding(embs1 + embs2)
        embs = np.array(embs, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return roots, embs, labels

    def __len__(self):
        return len(self.pairList)//self.batch_size

    def getData(self, treeName):
        root = self.treeList[treeName]
        emb = self.treeEmb[treeName]
        return root, emb


if __name__ == "__main__":
    pass
