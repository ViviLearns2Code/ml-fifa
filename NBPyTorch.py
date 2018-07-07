'''
    File name: NBPyTorch.py
    Author: vivilearnstocode
    Last modified: 06/10/2018
    Python version: 3.6
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class NBNet(nn.Module):
    def __init__(self,dim_in,h1,h2,dim_out,p_drop=0.2):
        super(NBNet, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.softplus = nn.Softplus() 
        self.fc1 = nn.Linear(dim_in,h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, dim_out)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))        
        x = self.dropout(x)
        x = self.fc3(x)   
        # link function: score = log(mu)
        muA = torch.exp(x[:,[0]])
        muB = torch.exp(x[:,[2]])
        # alpha (inverse of dispersion) > 0
        alphaA = self.softplus(x[:,[1]])
        alphaB = self.softplus(x[:,[3]])
        return (muA,alphaA,muB,alphaB)

class PoNet(nn.Module):
    def __init__(self,dim_in,h1,h2,dim_out,p_drop=0.2):
        super(PoNet, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.fc1 = nn.Linear(dim_in,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2, dim_out)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)   
        # link function: score = log(mu)     
        x = torch.exp(self.fc3(x))
        return x

class NBNLLLoss(nn.Module):
    def __init__(self,eps=1e-5,verbose=False):
        super(NBNLLLoss, self).__init__()
        self.eps = eps
        self.verbose = verbose

    def forward(self,mu,alpha,y):
        eps = self.eps
        theta = 1/(alpha+eps)
        s1 = -torch.lgamma(y+theta+eps)
        s2 = torch.lgamma(theta+eps)
        s3 = torch.lgamma(y+1)
        s4 = -theta*torch.log(theta+eps)
        s5 = -y*torch.log(mu+eps)
        s6 = (theta+y)*torch.log(theta+mu+eps)
        if self.verbose == True:
            print("mu",mu,mu.shape)
            print("theta",theta,theta.shape)
            print("s1",s1,s1.shape)
            print("s2",s2,s2.shape)
            print("s3",s3,s3.shape)
            print("s4",s4,s4.shape)
            print("s5",s5,s5.shape)
            print("s6",s6,s6.shape)
        NLL = torch.mean(s1+s2+s3+s4+s5+s6)
        if NLL < 0:
            raise Exception("NLL cannot be negative for PMF")
        return NLL     