import os
import torch
from torch.utils.data import Dataset, DataLoader

from treeBuilder import getChildrenList

def alignChildren(childrenList):
    '''
    childrenList: N x nodeNum x childrenNum
    '''
    maxNum = max([len(children) for s in childrenList for children in s])
    returnList = [[c+[0]*(maxNum-len(c)) for c in s] for s in childrenList] 
    return returnList


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
        embs = embs1 + embs2
        return roots, embs, labels

    def __len__(self):
        return len(self.treeList)//self.batch_size

    def getData(self, treeName):
        root = self.treeList[treeName]
        emb = self.treeEmb[treeName]
        return root, emb


if __name__ == "__main__":
    pass