import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable

class Lagrange_NET(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Lagrange_NET, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)
        torch.nn.init.xavier_uniform(self.linear1.weight)
        torch.nn.init.xavier_uniform(self.linear2.weight)
        torch.nn.init.xavier_uniform(self.linear3.weight)

        
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def derivatives(self, x):
        L = self.forward(x)
        dLdq = torch.autograd.grad(L, x, create_graph=True)
        return dLdq

# Train second NN to predict locations
from torch import autograd

class Q_NET(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Q_NET, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x