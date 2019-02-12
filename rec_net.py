import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from batching import batch


###########Simple RecNN architecture#############
class RecNN(nn.Module):
    def __init__(self, n_features, n_hidden,**kwargs):
        super().__init__()
        activation_string = 'relu'
        self.activation = getattr(F, activation_string)
        
        self.fc_u = nn.Linear(n_features, n_hidden)   ## W_u, b_u
        self.fc_h = nn.Linear(3 * n_hidden, n_hidden) ## W_h, b_h
        
        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_h.weight, gain=gain)

    def forward(self, jets):
        levels, children, n_inners, contents = batch(jets)
        n_levels = len(levels)
        embeddings = []
        
        for i, nodes in enumerate(levels[::-1]):
            j = n_levels - 1 - i
            inner = nodes[:n_inners[j]]
            outer = nodes[n_inners[j]:]
            u_k = self.fc_u(contents[j])
            u_k = self.activation(u_k) ##eq(3) in Louppe's paper
            
            if len(inner) > 0:
                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
                h_L = embeddings[-1][children[inner, zero]]
                h_R = embeddings[-1][children[inner, one]]
                
                h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
                h = self.fc_h(h)
                h = self.activation(h)
                embeddings.append(torch.cat((h, u_k[n_inners[j]:]), 0))
            else:
                embeddings.append(u_k)
        
        return embeddings[-1].view((len(jets), -1))

### Building fully connected classifier and pedictor###########
class Predict(nn.Module):
    def __init__(self, n_features, n_hidden, **kwargs):
        super().__init__()
        RecNN_transform=RecNN
        self.transform = RecNN_transform(n_features, n_hidden, **kwargs)
        activation_string = 'relu'
        self.activation = getattr(F, activation_string)
        
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)
        
        gain = nn.init.calculate_gain(activation_string)
        #########network initialization############
        nn.init.xavier_uniform(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform(self.fc3.weight, gain=gain)
        nn.init.constant(self.fc3.bias, 1)
    
    
    def forward(self, jets, **kwargs):
        out_stuff = self.transform(jets, **kwargs)
        h = self.fc1(out_stuff)
        h = self.activation(h)
        h = self.fc2(h)
        h = self.activation(h)
        h = F.sigmoid(self.fc3(h))
        return h

def square_error(y, y_pred):
    return (y - y_pred) ** 2

def log_loss(y, y_pred):
    return F.binary_cross_entropy(y_pred, y)


