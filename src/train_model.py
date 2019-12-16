import os
import numpy as np
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval_model(scores, annos):
    scores = np.array(scores)
    annos = np.array(annos).astype(int)
    annos[annos<0] = 0

    selected_thred = -1
    max_pred = 0
    for i in range(-9, 10):
        thred = i*0.1
        pred = (scores > thred).astype(int)
        tp = np.sum(pred*annos)
        fp = np.sum(pred) - tp
        fn = np.sum(annos) - tp
        recall = tp*1.0/(tp+fn)
        precision = tp*1.0/(tp+fp)
        if precision > max_pred:
            max_pred = precision
            selected_thred = thred
        print('thred {}, recall {}, prec {}'.format(round(thred,1), recall, precision))
    return thred


def prepareData(data_list_dir, dataset, mode, batch_size=1):
    funcList, pairList = loadDataList(data_list_dir, mode=mode)
    treeDict = {}
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
    dataLoader = DataLoader(dataset=dataset, shuffle=True)
    return dataLoader


def train():

    epoch_num = 10
    tbcnn = TBCNN(channels=[78, 600, 50]).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params=tbcnn.parameters(), lr=0.0002)

    # trainLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='train')
    # testLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='test')
    # valLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='val')

    trainLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='test')
    valLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', mode='val')

    for epoch_num in range(epoch_num):
        total_loss = 0
        for ind, data in enumerate(trainLoader):
            # if ind == 10000:
            #     break
            childrenList, emb, label = data

            childrenList = childrenList.squeeze(0).cuda()
            emb = emb.squeeze(0).cuda()
            label = label.squeeze(0).cuda()
            optimizer.zero_grad()

            output = tbcnn(childrenList, emb)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().item()
            if ind%50000 == 49999:
            # if ind%300 == 299:
                print("{} loss: {}".format(ind+1,total_loss/50000))
                total_loss = 0

        scores = []
        annos = []
        with torch.no_grad():
            for ind, data in enumerate(valLoader):
                if ind == 3000:
                    eval_model(scores, annos)
                    break

                childrenList, emb, label = data
                childrenList = childrenList.squeeze(0).cuda()
                emb = emb.squeeze(0).cuda()
                label = label.squeeze(0).cuda()
                output = tbcnn(childrenList, emb)
                label = label.cpu().tolist()
                output = output.cpu().tolist()
                annos.extend(label)
                scores.extend(output)


if __name__ == "__main__":
    train()
