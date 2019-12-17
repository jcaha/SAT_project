import os
import numpy as np
import numpy.random as npr
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
miniSet = True
miniRatio = 0.05

def f1_eval(scores, annos, thres):
    pred = (scores > thres).astype(int)
    tp = np.sum(pred*annos)
    fp = np.sum(pred) - tp
    fn = np.sum(annos) - tp
    recall = tp*1.0/(tp+fn)
    precision = tp*1.0/(tp+fp)
    f1 = recall*precision/(recall + precision)
    return recall, precision, f1


def eval_model(scores, annos, thres=None):
    scores = np.array(scores)
    annos = np.array(annos).astype(int)
    annos[annos<0] = 0

    selected_thres = -1
    max_f1 = 0
    if thres is None:
        for thres in range(-0.9, 0.9999, 0.1):
            recall, precision, f1 = f1_eval(scores, annos, thres)
            if f1 > max_f1:
                max_pred = precision
                selected_thres = thres
            print('val thres {}, recall {}, prec {}, f1 {}'.format(round(thres,1), recall, precision, thres))
    else:
        recall, precision, f1 = f1_eval(scores, annos, thres)
        print('test thres {}, recall {}, prec {}, f1 {}'.format(round(thres,1), recall, precision, thres))
    return thres

def prepareData(data_list_dir, dataset, wordEmb, mode, batch_size=1):
    funcList, rawPairList = loadDataList(data_list_dir, mode=mode)
    treeDict = {}
    print("------------------ start scan funcList ------------------")
    for item in funcList:
        funcPath = os.path.join(data_list_dir, dataset, item)
        root, nodeNum = getAstTree(funcPath)
        if nodeNum < 10 or nodeNum > 3000:
            continue
        treeDict[item] = root
    pairList = []
    for pair in rawPairList:
        root1, root2, label = pair.strip('\n').split('\t')
        if miniSet and mode == 'train' and npr.random() > miniRatio:
            continue
        elif miniSet and (mode == 'test' or mode == 'val') and npr.random() > miniRatio*5:
            continue
        elif root1 not in treeDict.keys() or root2 not in treeDict.keys():
            continue
        pairList.append(pair)

    print("------------------ start get treeEmbList ------------------")
    print(len(pairList))
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

    wordEmb = getEmbDict(os.path.join(data_list_dir, 'wordEmbedding.pkl'), datasetPath='../data/bigclonebenchdata')
    trainLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', wordEmb=wordEmb, mode='train')
    testLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', wordEmb=wordEmb, mode='test')
    valLoader = prepareData(data_list_dir, dataset='bigclonebenchdata', wordEmb=wordEmb, mode='val')

    for epoch_num in range(epoch_num):
        total_loss = 0
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
            total_loss += loss.cpu().detach().item()
            if ind%5000 == 4999:
                print("{} loss: {}".format(ind+1,total_loss/5000))
                total_loss = 0

        with torch.no_grad():
            scores = []
            annos = []
            for ind, data in enumerate(valLoader):
                childrenList, emb, label = data
                childrenList = childrenList.squeeze(0).cuda()
                emb = emb.squeeze(0).cuda()
                label = label.squeeze(0).cuda()
                output = tbcnn(childrenList, emb)
                label = label.cpu().tolist()
                output = output.cpu().tolist()
                annos.extend(label)
                scores.extend(output)
            thres = eval_model(scores, annos)

            scores = []
            annos = []
            for ind, data in enumerate(testLoader):
                childrenList, emb, label = data
                childrenList = childrenList.squeeze(0).cuda()
                emb = emb.squeeze(0).cuda()
                label = label.squeeze(0).cuda()
                output = tbcnn(childrenList, emb)
                label = label.cpu().tolist()
                output = output.cpu().tolist()
                annos.extend(label)
                scores.extend(output)
            eval_model(scores, annos, thres)


if __name__ == "__main__":
    train()
