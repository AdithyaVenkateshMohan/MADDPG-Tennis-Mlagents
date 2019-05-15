import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        if actor:
            self.bnS = nn.BatchNorm1d(num_features = input_dim)
        else:
            self.bnS = nn.BatchNorm1d(num_features = input_dim - 4)
            
        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.bn1 = nn.BatchNorm1d(num_features = hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.nonlin = f.relu #leaky_relu
        self.leaky = f.relu
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)

    def forward(self, x , batch = True):
        if self.actor:
            # return a vector of the force
            if(batch):
                out1 = self.bnS(x)
            else:
                out1 = x
                
            out1 = self.fc1(out1)
            h1 = self.leaky(out1)
            h2 = self.leaky(self.fc2(h1))
            h3 = (self.fc3(h2))
            
            norm = torch.tanh(h3)
            
            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            return norm
        else:
            print("error don't pass critic here")
            return 0
    def critic_forward(self, state, action , batch = True):
        if not self.actor:
            out1 = self.bnS(state)
            out1 = torch.cat((out1, action),dim=1)
            out1 = self.fc1(out1)
            h1 = self.leaky(out1)
            h2 = self.leaky(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3
        else:
            print("error don't pass actor here")
            return 0
            