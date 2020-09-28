"""Problem 3 - Training on MNIST"""
import numpy as np
from mytorch.tensor import Tensor
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.nn.linear import Linear
from mytorch.nn.batchnorm import BatchNorm1d 
from mytorch.nn.activations import ReLU
from mytorch.nn.loss import CrossEntropyLoss

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)
# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion

    # TODO: Call training routine (make sure to write it below)
    l=[Linear(784,20),BatchNorm1d(20),ReLU(),
                       Linear(20,10)]
    model=Sequential(*l)
    criterion=CrossEntropyLoss()
    optimizer= SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3)
    
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    train_loss=0
    val_accuracies = []
    #print("Num_epochs is:",num_epochs)
    for epoch in range(num_epochs):
        print("Epoch:",epoch)
        model.train()
        
        train_y=np.reshape(train_y,(train_y.shape[0],1))
        batch=np.hstack((train_x,train_y))
        
        batches=np.split(batch,550)#
        
        for batch_idx,x in enumerate(batches): 
            y=x[:,-1]
            x=x[:,:-1]
            output=model(Tensor(x))
            loss=criterion(output,Tensor(y))
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
            train_loss+=loss.data  
            #print("Training Loss :",train_loss/train_y.shape[0]) 
            if(batch_idx%100==1):
                accuracy=validate(model, val_x, val_y,criterion)
                val_accuracies.append(accuracy)
                #print("Validation accuracy at Batchidx {} is : {}".format(batch_idx,val_accuracies[-1]))
                model.train()
        
    # TODO: Implement me! (Pseudocode on writeup)
    return val_accuracies

def validate(model, val_x, val_y,criterion):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    model.eval() 
    valid_loss=0
    total_accuracy=0
    val_y=np.reshape(val_y,(val_y.shape[0],1))
    batch=np.hstack((val_x,val_y))
    
    batches=np.split(batch,50)
    
    for batch_idx,x in enumerate(batches):
            accuracy=0
            y=x[:,-1]
            x=x[:,:-1]
            output=model(Tensor(x))
            
            predictions=np.argmax(output.data,axis=1)
            loss=criterion(output,Tensor(y))
            valid_loss+=loss.data
            y=Tensor(y)
            #print("Shape of y:",y.shape)
            for i in range(0,y.shape[0]):
                #print('y :{} and pred :{}'.format(int(y.data[i]),predictions[i]))
                if int(y.data[i])==predictions[i]:
                    accuracy+=1
            #print("Accuracy is:",(accuracy/y.shape[0])*100)
            total_accuracy+=accuracy
            #print("Validation loss:",valid_loss/val_y.shape[0])
    #TODO: implement validation based on pseudocode
    return (total_accuracy/len(val_x))*100
