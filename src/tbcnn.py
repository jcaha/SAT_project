import torch
import torch.nn as nn
import torchvision
import math
from utils import gatherFeature, convWeights


class TBCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert len(channels) == 3
        self.channels = channels
        self.tbconv = nn.Sequential(
            nn.Linear(in_features=3*self.channels[0], out_features=self.channels[1], bias=True),
            nn.Tanh()
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels[1], out_features=self.channels[2], bias=True),
            nn.Tanh()
        )
        self.cosin_sim = nn.CosineSimilarity()
        self.init_params()

    def forward(self, childrenList, nodeEmb):
        batch_size, max_node_num, max_children_num = childrenList.shape
        feature = gatherFeature(childrenList, nodeEmb)  # feature: (batch_size, max_node_num, max_children_num+1, feature)
        weights = convWeights(childrenList)  # (batch_size, max_node_num, max_children_num+1, 3)
        # print(feature.dtype, weights.dtype)
        weightedFeature = torch.matmul(weights.transpose(2,3), feature)  # weight of Ws: (batch_size, max_node_num, feature, 3)
        weightedFeature = weightedFeature.reshape(batch_size, max_node_num, -1)  # (batch_size, max_node_num, feature*3)
        output = self.tbconv(weightedFeature)  # (batch_size, max_node_num, channel)
        output = self.pool_tree(output) # (batch_size, channel)
        output = self.linear(output)    # shape: (batch_size, feature)
        batch_size = batch_size//2
        vector1 = output[:batch_size]
        vector2 = output[batch_size:]
        similarity = self.cosin_sim(vector1, vector2)
        return similarity

    def pool_tree(self, treeFeature):
        output, _ = torch.max(treeFeature, dim=1)
        return output

    def init_params(self):
        std_tbconv = 1.0/math.sqrt(self.channels[0])
        std_linear = 1.0/math.sqrt(self.channels[1])
        for layer in self.tbconv.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=std_tbconv)
                nn.init.normal_(layer.bias, std=2*std_tbconv)
        for layer in self.tbconv.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=std_linear)
                nn.init.normal_(layer.bias, std=2*std_linear)


if __name__ == "__main__":
    pass
