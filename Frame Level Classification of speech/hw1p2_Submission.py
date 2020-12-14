#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:38:18 2020

@author: shayereesarkar
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import random
import copy
import time
import torch.nn.functional as F
import pandas as pd
from pandas import DataFrame

trainset =np.load('/home/ironman/shayeree/records/hw1p2/train.npy',allow_pickle=True)
testset=np.load('/home/ironman/shayeree/records/hw1p2/test.npy',allow_pickle=True)
valid=np.load('/home/ironman/shayeree/records/hw1p2/dev.npy',allow_pickle=True)
train_labels=np.load('/home/ironman/shayeree/records/hw1p2/train_labels.npy',allow_pickle=True)
valid_labels=np.load('/home/ironman/shayeree/records/hw1p2/dev_labels.npy',allow_pickle=True)

context=22
num_epochs=2


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,X,Y,context):
        self.X=copy.deepcopy(X)     
        self.Y=copy.deepcopy(Y)
        self.index_ij=[]
        self.cs=context
        self.row=0
           
        for i in range(0,len(self.X)): 
            self.row+=self.X[i].shape[0]
            for j in range(self.X[i].shape[0]):
                self.index_ij.append((i,j))#This is the tuple for the lists 
                
            self.X[i]=np.pad(self.X[i],pad_width=((context,context),(0,0)),mode='constant',constant_values=0)
            self.Y[i]=np.pad(self.Y[i],pad_width=((context,context)),mode='constant',constant_values=0)             
        
    def __len__(self):
        return self.row#This will look up all the samples in the 22000 recordings
        
    def __getitem__(self,index): 
        
        (index1,index2)=self.index_ij[index]
        
        X=self.X[index1][index2:index2+2*self.cs+1].reshape(-1)
        Y=self.Y[index1][index2+self.cs].reshape(-1)
        #k=index2+2*self.cs+1
        
        return X,Y 



train_dataset=MyDataset(trainset,train_labels,context)
valid_dataset=MyDataset(valid,valid_labels,context)
train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=4096,num_workers=36,pin_memory=True)
valid_loader=torch.utils.data.DataLoader(valid_dataset,shuffle=False,batch_size=2048,num_workers=36,pin_memory=True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        layers=[nn.Linear(13*(2*context+1),1024),
                       nn.BatchNorm1d(1024),
                       nn.GELU(),
                       nn.Linear(1024,2048),
                       nn.BatchNorm1d(2048),
                       nn.GELU(),
                       nn.Linear(2048,4096),
                       nn.BatchNorm1d(4096),
                       nn.GELU(),
                       nn.Linear(4096,2048),
                       nn.BatchNorm1d(2048),
                       nn.GELU(),
                       nn.Linear(2048,1024),
                       nn.BatchNorm1d(1024),
                       nn.GELU(),
                       nn.Linear(1024,512),
                       nn.BatchNorm1d(512),
                       nn.GELU(),
                       nn.Linear(512,346)]
        
        self.layers=nn.Sequential(*layers)# * operator does.Essentially, it opens up the list and directly and puts them in as arguments of nn.Sequential.
    def forward(self,x):
        out=self.layers(x) 
        return out
    
model=Model()
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model,device_ids=[3]).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

def train(num_epochs,train_loader,valid_loader):
    
    for epoch in range(num_epochs):
        model.train()
        #print(' epoch :',epoch+1)
        train_loss=0
        valid_loss=0
        for batch_idx,(x,y) in enumerate(train_loader):
            start=time.time()
            x=x.to(device)
            y=y.to(device)
            y=y.reshape(-1)

            output=model(x.float())
            loss=criterion(output,y)
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()

            train_loss+=loss.item()
            stop=time.time()
            print('E: %d, B: %d / %d, Train Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f' % (epoch+1, batch_idx+1, 
                                                                           len(train_loader),
                  loss.item(),train_loss/(batch_idx+1),stop-start),end='\n ')
            
            if(epoch==num_epochs-1):
                torch.save(model.state_dict(), "model_gelu.pth")
            del loss
            del y
            del x

        model.eval() 
        print("In evaluation -->")
        total_accuracy=0
        total=0

        for batch_idx,(x,y) in enumerate(valid_loader):
            accuracy=0
            x=x.to(device)
            y=y.to(device)
            y=y.reshape(-1)
            output=model(x.float())
            loss=criterion(output,y)
            valid_loss+=loss.item()
            #We have to calculate accuracy
            predictions=F.softmax(output,dim=1)
            _,top_prediction=torch.max(predictions,1)#To get the 

            top_pred_labels=top_prediction.view(-1)
            accuracy+=torch.sum(torch.eq(top_pred_labels,y)).item()
            total+=len(y)
            total_accuracy+=accuracy
            print('E: %d, B: %d / %d, Valid Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f, Validation Accuracy : %.4f' % (epoch+1, batch_idx+1, 
                                                                           len(valid_loader),
                  loss.item(),valid_loss/(batch_idx+1),stop-start,(accuracy*100)/len(y)),end='\n ')
            del x
            del y
        model.train()    
        print(" Validation Accuracy is {} at Epoch {} ".format((total_accuracy*100)/total ,epoch+1))
        
train(num_epochs,train_loader,valid_loader)


class MytestDataset(torch.utils.data.Dataset):
    def __init__(self,X,context):
        self.X=copy.deepcopy(X)  
        self.index_ij=[]
        self.row=0
        self.cs=context
           
        for i in range(0,len(self.X)): 
            self.row+=self.X[i].shape[0]
            for j in range(self.X[i].shape[0]):
                self.index_ij.append((i,j))#This is the tuple for the lists            
            self.X[i]=np.pad(self.X[i],pad_width=((context,context),(0,0)),mode='constant',constant_values=0)
    def __len__(self):
        return self.row#This will look up all the samples in the 22000 recordings
        
    def __getitem__(self,index): 
        
        (index1,index2)=self.index_ij[index]
        #print('Index 1 :{} and Index 2:{}'.format(index1,index2))
        X=self.X[index1][index2:index2+2*self.cs+1].reshape(-1)
        #print(X.shape)
        
        return X

testset =np.load('/home/ironman/shayeree/records/hw1p2/test.npy',allow_pickle=True)
test_dataset=MytestDataset(testset,context)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=2048,num_workers=4,pin_memory=True)
sample_csv=pd.read_csv('/home/ironman/shayeree/records/hw1p2/sample.csv')

def test_classify(test_loader):
    l=[]
    
    for batch_idx,(x) in enumerate(test_loader):
        x=x.to(device)
        output=model(x.float())
        predictions=F.softmax(output,dim=1)
        _,top_prediction=torch.max(predictions,1)#To get the 
        top_pred_labels=top_prediction.view(-1)
        
        for i in top_pred_labels:
            l.append(i.item())
        #print(len(l))   
        #sample_csv.loc[batch_idx,'label']=top_pred_labels.item()     
    return l     

labels=test_classify(test_loader)
sample_new = DataFrame(labels,columns=['label'])       
sample_new.insert(0, 'id',list(range(0, 1593223)))

sample_new.to_csv('my_submission_10.csv',index=False) 
    