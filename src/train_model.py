import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from AstEmbLoader import AstEmbDataset
from tbcnn import TBCNN
from treeBuilder import getAstTree
from getEm import getEmbDict, getDataList
from utils import loadDataList

data_list_dir = '../data'
dataset_path = '../data/bigclonebenchdata'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prepareData(data_list_dir, dataset, mode, batch_size=1):
    funcList, pairList = loadDataList(data_list_dir, mode=mode)
    treeDict = {}
    treeEmbDict = {}
    treeList = []
    print("------------------ start scan funcList ------------------")
    for item in funcList:
        funcPath = os.path.join(data_list_dir, dataset, item)
        root, nodeNum = getAstTree(funcPath)
        treeDict[item] = root
    wordEmb = getEmbDict(os.path.join(data_list_dir, 'wordEmbedding.pkl'), treeDict)
    print("------------------ start get treeEmbList ------------------")
    childrenListDict, treeEmb = getDataList(treeDict, wordEmb)
    # if mode=='train':
    #     pairList = pairList[:10000]
    dataset = AstEmbDataset(pairList=pairList, treeList=childrenListDict, treeEmb=treeEmb, batch_size=batch_size)
    dataLoader = DataLoader(dataset=dataset)
    return dataLoader


def train():

    epoch_num = 10
    tbcnn = TBCNN(channels=[78, 600, 50]).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params=tbcnn.parameters(), lr=0.0002)

    trainLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='train')
    testLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='test')
    valLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='val')

    # trainLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='test')

    for epoch_num in range(epoch_num):
        for ind, data in enumerate(trainLoader):
            childrenList, emb, label = data
            childrenList = childrenList.squeeze(0).cuda()
            emb = emb.squeeze(0).cuda()
            label = label.squeeze(0).cuda()
            optimizer.zero_grad()

            output = tbcnn(childrenList, emb)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if ind%100 ==99:
                print("{} loss: {}".format(ind+1,loss))


if __name__ == "__main__":
    train()