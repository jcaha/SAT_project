import os
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from AstEmbLoader import AstEmbDataset, alignChildren, alignEmbedding
from tbcnn import TBCNN
from treeBuilder import getAstTree, getChildrenList, getAstNodeList
from getEm import getEmbDict, getDataList, getCharEmbedding, getWordEmbedding
from utils import loadDataList

test_dir = '../data/test_dir'
model_path = '../data/models'
model = 'tbcnn_batch8'
epoch = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# thres = 0.5
# func1_name = '116038.txt'
# func2_name = '3874060.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--f1", help="func1")
parser.add_argument("--f2", help="func1")
parser.add_argument("-t", type=float, default=0.5, help="threshold")
args = parser.parse_args()

func1_name = args.f1
func2_name = args.f2
thres = args.t

def process_func(func_path, charEmb):
    root, num = getAstTree(func_path)
    childrenList, wordList = getChildrenList(root)
    wordSet = getAstNodeList(root)
    embDict = {}
    wordEmbList = []
    for word in wordSet:
        embDict[word] = getWordEmbedding(word, charEmb)
    for word in wordList:
        wordEmbList.append(embDict[word])
    return childrenList, wordEmbList



if __name__ == "__main__":
    load_model = '{}_{}.pth'.format(model, epoch)
    model_file = os.path.join(model_path, load_model)
    device = torch.device("cuda")
    tbcnn = TBCNN(channels=[78, 600, 50]).to(device)
    params = torch.load(model_file, map_location=device)
    tbcnn.load_state_dict(params)
    print('load model from epoch {} of {}.'.format(epoch, model))

    charEmbedding = getCharEmbedding()
    func1_name = os.path.join(test_dir, func1_name)
    func2_name = os.path.join(test_dir, func2_name)
    cList1, wordEmb1 = process_func(func1_name, charEmbedding)
    cList2, wordEmb2 = process_func(func2_name, charEmbedding)
    cLists = [cList1, cList2]
    wordEmbs = [wordEmb1, wordEmb2]
    cLists = torch.from_numpy(np.array(alignChildren(cLists))).cuda()
    wordEmbs = torch.from_numpy(np.array(alignEmbedding(wordEmbs),dtype=np.float32)).cuda()
    output = tbcnn(cLists, wordEmbs)
    output = output.detach().cpu().item()
    print('confidence is {} with threshold {}'.format(output, thres))
    if output > thres:
        print('positive.')
    else:
        print('negative.')
