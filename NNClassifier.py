'''
    File name: NNClassifier.py
    Author: vivilearnstocode
    Last modified: 07/07/2018
    Python version: 3.6
'''

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import scipy as sc
import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,dim_input,dim_output,dropout=0.2):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(dim_input, 20)
        self.dropout = nn.Dropout(p=dropout)
        self.lin2 = nn.Linear(20, dim_output)

    def forward(self, x):
        x = nn.functional.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x

class NNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,input_dim,dropout=0.2,classes=3,iter=100,lr=0.01,alpha=0.99,momentum=0,verbose=False):
        self.lr = lr
        self.alpha = alpha
        self.momentum = momentum
        self.dropout = dropout
        self.iter = iter
        self.verbose = verbose  
        self.input_dim = input_dim      
        self.clf = Net(dim_input=input_dim,dropout=dropout,dim_output=classes)
        self.classes = classes
            
    def fit(self,X,y):
        # convert to variable
        X_train = torch.autograd.Variable(torch.from_numpy(X).float())
        y_train = torch.autograd.Variable(torch.from_numpy(y).long())
        # optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.clf.parameters(),lr=self.lr, alpha=self.alpha, momentum=self.momentum)

        for t in range(self.iter):
            scores = self.clf(X_train)
            scores_np = scores.data.numpy()
            norm = scores_np.max(axis=1,keepdims=True)
            scores_np = scores_np-norm
            proba = np.exp(scores_np)/np.sum(np.exp(scores_np),axis=1,keepdims=True)
            y_pred = np.argmax(proba,axis=1)
            loss = criterion(scores, y_train)
            if self.verbose == True:
                train_acc = np.sum(y==y_pred)/y_pred.shape[0]
                print("step",t,"loss",loss.data[0],"training acc",train_acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def predict(self,X):
        X_test = torch.autograd.Variable(torch.from_numpy(X).float())
        scores_np = self.clf.forward(X_test).data.numpy()
        norm = scores_np.max(axis=1,keepdims=True)
        scores_np = scores_np-norm
        proba = np.exp(scores_np)/np.sum(np.exp(scores_np),axis=1,keepdims=True)
        y_pred = np.argmax(proba,axis=1)
        return y_pred

    def predict_proba(self,X):
        X_test = torch.autograd.Variable(torch.from_numpy(X).float())
        scores_np = self.clf.forward(X_test).data.numpy()    
        norm = scores_np.max(axis=1,keepdims=True)
        scores_np = scores_np-norm     
        proba=np.exp(scores_np)/np.sum(np.exp(scores_np),axis=1, keepdims=True)
        return proba

    def get_params(self,deep=True):
        return {
            "classes": self.classes,
            "input_dim": self.input_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "alpha": self.alpha,
            "momentum": self.momentum,
            "iter": self.iter,
            "verbose": self.verbose
        }