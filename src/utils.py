import os
import torch


def loadDataset(setpath, file_name):
    setpath = os.path.join(setpath, file_name)
    with open(setpath, 'r') as f:
        data = f.read()
        if "Func" in file_name:
            data = data.strip('\t').split('\t')
        else:
            data = data.strip('\n').split('\n')
    return data


def loadDataList(datapath, mode):
    funclist = loadDataset(datapath, '{}FuncList.txt'.format(mode))
    pairlist = loadDataset(datapath, '{}Pair.txt'.format(mode))
    return funclist, pairlist


def gatherFeature(children, nodeEmb):
    '''
    input:
        children: (batch_size, max_node_num, max_children_num)
        nodeEmb: (batch_size, max_node_num, feature_size)
    output:
        treeFeature: (batch_size, max_node_num, max_children_num+1, feature_size)
    '''
    batch_size, max_node_num, max_children_num = children.shape
    feature_size = nodeEmb.shape[-1]
    parentNode = torch.arange(0, max_node_num, dtype=torch.int64)
    parentNode = parentNode.view(1,max_node_num,1).repeat(batch_size,1,1).to(children.device)  # (batch_size, max_node_num, 1)
    children = torch.cat([parentNode, children], dim=2)  # (batch_size, max_node_num, max_children_num+1)
    feature_index = children.unsqueeze(-1).expand(-1, -1, -1, feature_size)    # (batch_size, max_node_num, max_children_num+1, feature_size)

    nodeEmb = nodeEmb.unsqueeze(1)
    nodeEmb = nodeEmb.expand(-1,max_node_num,-1,-1)
    treeFeature = torch.gather(nodeEmb, dim=2, index=feature_index)
    return treeFeature

def convWeights(children):
    '''
    input:
        children: (batch_size, max_node_num, max_children_num)
    output:
        convWeights: (batch_size, max_node_num, max_children_num+1, 3)
    '''
    batch_size, max_node_num, max_children_num = children.shape
    children = children.float()
    device = children.device
    topWeights = torch.ones(batch_size, max_node_num, 1).to(device)
    topWeights = torch.cat([topWeights, torch.zeros_like(children)], dim=2) # (batch_size, max_node_num, max_children_num+1)
    children_mask = (children>0).to(device)
    childrenCnt = torch.sum(children_mask, dim=2, keepdim=True).expand(-1, -1, max_children_num+1)  # (batch_size, max_node_num, max_children_num+1)
    children_mask = torch.cat([torch.zeros(batch_size, max_node_num,1,device=device), children_mask.float()], dim=2)  # (batch_size, max_node_num, max_children_num+1)
    rightWeight = torch.arange(-1, max_children_num, dtype=torch.float32).reshape(1,1,-1).repeat(batch_size, max_node_num, 1).to(device)
    oneChild = torch.cat([
        torch.zeros(batch_size, max_node_num, 1),
        torch.empty(batch_size, max_node_num, 1).fill_(0.5),
        torch.zeros(batch_size, max_node_num, max_children_num-1)
    ], dim=2).to(device)
    # print(childrenCnt.equal==torch.zeros_like(childrenCnt))
    rightWeight = torch.where(childrenCnt==1, oneChild, torch.div(torch.mul(rightWeight,children_mask.float()),childrenCnt.float()-1))  # (batch_size, max_node_num, max_children_num+1)
    leftWeitght = torch.mul(children_mask.float(),1 - rightWeight) # (batch_size, max_node_num, max_children_num+1)
    # leftWeitght[:,:,0].fill_(0)
    convWeights = torch.cat([
        topWeights.unsqueeze(-1),
        leftWeitght.unsqueeze(-1),
        rightWeight.unsqueeze(-1)
    ], dim=3)
    return convWeights

if __name__ == "__main__":
    '''
    input:
        children: (batch_size, max_node_num, max_children_num)
    output:
        convWeights: (batch_size, max_node_num, max_children_num+1, 3)
    '''
    # convWeights
    # nodeTensor = torch.tensor([[
    #     [1,2,3],
    #     [6,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [4,5,0],
    # ]])
    # print(convWeights(nodeTensor))  # 1x7x3 -> 1x7x4x3

    '''
    input:
        children: (batch_size, max_node_num, max_children_num)
        nodeEmb: (batch_size, max_node_num, feature_size)
    output:
        treeFeature: (batch_size, max_node_num, max_children_num+1, feature_size)
    '''
    # gatherFeature
    # nodeTensor = torch.tensor([[    # 1x7x3
    #     [1,2,3],
    #     [6,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [4,5,0],
    # ]])
    # feature = torch.zeros(1,7,7)
    # for i in range(7):
    #     feature[:,i,i] = 1
    # print(gatherFeature(nodeTensor, feature))   # 1x7x3 -> 1x7x4x7

     
    # childrenList = torch.tensor([[
    #     [1,2,3],
    #     [6,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [0,0,0],
    #     [4,5,0],
    # ]])
    # nodeEmb = torch.zeros(1,7,7)
    # for i in range(7):
    #     nodeEmb[:,i,i] = 1
    # feature = gatherFeature(childrenList, nodeEmb)  # feature: (batch_size, max_node_num, max_children_num+1, feature)
    # weights = convWeights(childrenList)  # (batch_size, max_node_num, max_children_num+1, 3)
    # print(weights.transpose(2,3).shape)
    # print(feature.shape)
    # weightedFeature = torch.matmul(weights.transpose(2,3), feature)  # weight of Ws: (batch_size, max_node_num, feature, 3)
    # weightedFeature = weightedFeature.reshape(1, 7, -1)  # (batch_size, max_node_num, feature*3)
    # print(weightedFeature)  # (1, 7, 21)
