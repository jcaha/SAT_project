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
    rightWeight = torch.where(childrenCnt==0, oneChild, torch.div(torch.mul(rightWeight,children_mask.float()),childrenCnt.float()))  # (batch_size, max_node_num, max_children_num+1)
    leftWeitght = 1 - rightWeight
    convWeights = torch.cat([
        topWeights.unsqueeze(-1),
        leftWeitght.unsqueeze(-1),
        rightWeight.unsqueeze(-1)
    ], dim=3)
    return convWeights


if __name__ == "__main__":
    pass
