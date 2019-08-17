import torch.nn as nn
import torch
import numpy as np
"""
Deep Differentiable Decision Tree 
Last Updated 8/17/19
"""
def last_node(layer):
    if layer == 0:
        return 0
    else:
        return last_node(layer-1)+2**layer

class Node(nn.Module):
    def __init__(self,input_size,output_size, alpha,use_gpu,vectorized=False, is_leaf=False):
        super(Node, self).__init__()
        self.comparators = None
        self.set_comps = True
        self.vectorized= vectorized
        self.use_gpu = use_gpu
        self.calcLayer = nn.Linear(input_size, int(output_size))
        if self.vectorized:
            probWeights = torch.rand(input_size, requires_grad=True)
            if self.use_gpu:
                probWeights = probWeights.cuda()
            self.probWeights = nn.Parameter(probWeights)
            self.attnLayer = nn.Linear(input_size, input_size)
        else:
            self.probLayer = nn.Linear(input_size,1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.alpha = alpha
        if not is_leaf:
            self.out = None
        else:
            self.out = nn.Parameter(torch.rand(output_size,requires_grad=True))
        self.probab =None
        self.softmax = nn.Softmax(dim=-1)
    def res(self,input_data):
        return self.relu((self.calcLayer(input_data)))
    def prob(self,input_data):
        if self.vectorized:
            if self.set_comps:
                    avg_bias = torch.mul(input_data, self.probWeights)
                    avg_bias = torch.mean(avg_bias, dim=0)
                    avg_bias = torch.mul(-1,avg_bias)
                    if self.use_gpu:
                        avg_bias = avg_bias.cuda()
                    self.comparators = nn.Parameter(avg_bias, requires_grad=True)
            dist = torch.mul(self.probWeights, input_data)
            dist = torch.add(dist, self.comparators)
            dist = self.sig(torch.mul(self.alpha,dist))
            attn = self.softmax(self.attnLayer(input_data))
            attX = torch.mul(dist,attn)
            return torch.sum(attX, dim=1).view(-1,1)
        else:
            if self.set_comps:
                avg_bias = torch.sum(torch.mul(input_data, self.probLayer.weight), dim=1)
                avg_bias = torch.mean(avg_bias)
                avg_bias = torch.mul(-1,avg_bias)
                if self.use_gpu:
                    avg_bias = avg_bias.cuda()
                self.comparators = nn.Parameter(avg_bias, requires_grad=True)
            self.probLayer.bias = self.comparators
            return self.sig(self.alpha*(self.probLayer(input_data)))
class DeepDDT(nn.Module):
    def __init__(self,
                 input_dim,nodes,
                 leaves,
                 output_dim=None,
                 alpha=1.0,
                 is_value=False,
                 use_gpu=False,
                 vectorized=True, train_alpha=False):
        super(DeepDDT, self).__init__()


        self.use_gpu = use_gpu
        self.vectorized = vectorized
        self.leaf_init_information = leaves
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self.selector = None
        self.nodes = None
        self.alpha = None

        self.init_alpha(alpha, train_alpha=train_alpha)
        self.init_nodes(nodes)

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.is_value = is_value

    def init_nodes(self, nodes):
        nodes= [Node(input_size=self.input_dim, output_size=self.input_dim, alpha=self.alpha, vectorized=self.vectorized, use_gpu=self.use_gpu)]
        if type(self.leaf_init_information) is int:
            depth = int(np.floor(np.log2(self.leaf_init_information)))
        else:
            depth = 4
        for level in range(1, depth):
            for node in range(2**level):
                nodes.append(Node(input_size=self.input_dim*2, output_size=self.input_dim, alpha = self.alpha, vectorized=self.vectorized,use_gpu=self.use_gpu))
        for leaf in range(2**depth):
            nodes.append(Node(input_size=self.input_dim, output_size=self.output_dim, alpha = self.alpha, vectorized=self.vectorized, use_gpu=self.use_gpu, is_leaf=True))
        self.nodes = nn.ModuleList(nodes)

    def init_alpha(self, alpha, train_alpha):
        alpha = torch.Tensor([alpha])
        if self.use_gpu:
            alpha = alpha.cuda()
        alpha.requires_grad = True
        self.alpha = alpha
        if(train_alpha):
            self.alpha = nn.Parameter(alpha)


    def forward(self, input_data, embedding_list=None):
        batch_size= len(input_data)
        if type(self.leaf_init_information) is int:
            depth = int(np.floor(np.log2(self.leaf_init_information)))
        else:
            depth = 4
        leaf_num = 2**depth

        output = torch.tensor([],dtype=torch.float, requires_grad=True)
        if (self.use_gpu):
            output = output.cuda()
        self.nodes[0].out = self.nodes[0].res(input_data)
        prob = self.nodes[0].prob(input_data)
        self.nodes[1].probab = 1-prob
        self.nodes[2].probab = prob
        self.nodes[0].set_comps = False
        for node in range(1, leaf_num-1):
            self.nodes[node].out = self.nodes[node].res(torch.cat([self.nodes[(node-1)//2].out,input_data],1))
            prob = self.nodes[node].prob(torch.cat([self.nodes[(node-1)//2].out,input_data],1))
            self.nodes[2*node + 1].probab = torch.mul(self.nodes[node].probab,1-prob)
            self.nodes[2*node+2].probab = torch.mul(self.nodes[node].probab,prob)
            self.nodes[node].set_comps = False


        for i in range(0,2**depth):
            output = torch.cat([output, self.nodes[i+(2**depth)-1].probab.mul(self.nodes[i+(2**depth)-1].out)],dim=1)
        output = output.view(batch_size,leaf_num,self.output_dim)
        actions = output.sum(dim=1)
        if not self.is_value:
            return self.softmax(actions)
        else:
            return actions
