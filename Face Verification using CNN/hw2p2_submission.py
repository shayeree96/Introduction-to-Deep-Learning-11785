import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import sklearn.metrics
from torchvision import datasets, transforms
from PIL import Image
import time
import torch.nn.functional as F

class MyDataset():
    def __init__(self,directory,transform):
        
        self.X=[]    
        self.Y=[]
        self.transform=transform
        
        sub_folders=os.listdir(directory)
        #print(sub_folders)
        
        for i in range(len(sub_folders)):
            
            img_dir=os.listdir(os.path.join(directory,sub_folders[i]))
            for img in range(len(img_dir)):
                self.Y.append(i)
                self.X.append(os.path.join(os.path.join(directory,sub_folders[i]),img_dir[img]))  

                
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,index): 
        
        img=Image.open(self.X[index])
        img=self.transform(img)
        
        return img,self.Y[index]

#We have to find the labels
train_dir='/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/classification_data/train_data'
valid_dir='/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/classification_data/val_data'
test_dir='/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/classification_data/test_data'

transform = transform = transforms.Compose([#transforms.RandomResizedCrop(224),
                                                    #transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                                                                ])


train_dataset=MyDataset(train_dir,transform)
valid_dataset=MyDataset(valid_dir,transform)
test_dataset=MyDataset(test_dir,transform)

train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=128,num_workers=36,pin_memory=True)
valid_loader=torch.utils.data.DataLoader(valid_dataset,shuffle=False,batch_size=64,num_workers=36,pin_memory=True)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=64,num_workers=36,pin_memory=True)

#Building the Model
class BasicBlock(nn.Module):
    exp = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.skip = nn.Sequential()
        if stride != 1 or in_channel!= self.exp*out_channel:
            self.skip = nn.Sequential(nn.Conv2d(in_channel, self.exp*out_channel,
                                                    kernel_size=1, stride=stride, bias=False),
                                                    nn.BatchNorm2d(self.exp*out_channel))

    def forward(self, x):
        output = F.relu(self.bn_1(self.conv_1(x)))
        output = self.bn_2(self.conv_2(output))
        output += self.skip(x)# we add the skip connection
        output = F.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4000):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.exp, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.exp
        return nn.Sequential(*layers)

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output  = self.layer1(output)
        output  = self.layer2(output)
        output  = self.layer3(output)
        output  = self.layer4(output)
        output  = self.avgpool(output)
        output  = output.view(output.size(0), -1)
        output  = self.linear(output)
        return output

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

#Assigning the model
model=ResNet34()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#checkpoint = torch.load('hw2p2_checkpoint_Resnet34_exp2.pth')#Resnet_50_checkpoint
model = nn.DataParallel(model,device_ids=[0,1,5,7]).to(device)
#model.load_state_dict(checkpoint)


#Now we take the images from the verification dataset
test_list=np.loadtxt('/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/verification_pairs_test.txt',delimiter=' ',dtype='str')
valid_list=np.loadtxt('/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/verification_pairs_val.txt',delimiter=' ',dtype='str')

#Create a dataset class to read the valid list
class Custom_dataset():
    def __init__(self,path,transform):
        self.img1=path[:,0]
        self.img2=path[:,1]
        self.Y=path[:,2]
        self.transform=transform
             
    def __len__(self):
        return self.img1.shape[0]
    
    def __getitem__(self,index): 
        
        dir='/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2'
        img1=Image.open(os.path.join(dir,self.img1[index]))
        img2=Image.open(os.path.join(dir,self.img2[index]))
        img1=self.transform(img1)
        img2=self.transform(img2)
        
        return img1,img2,torch.tensor(int(self.Y[index]))


valid_set=Custom_dataset(valid_list,transform)
valid_ver_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=128, shuffle = True, num_workers=16)

#Create functions to calculate cosine similarity and generate scores on the pairwise valid list
def predict_score(model, image_1, image_2):   
    
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    scores=[]
        
    embeddings_1 = model(image_1)
    embeddings_2 = model(image_2)

    for i in range(len(embeddings_1)):         
        scores.append(torch.Tensor(cos(embeddings_1[i].detach().cpu(), embeddings_2[i].detach().cpu()).numpy())) 
       
    return scores


def score_generator(data_loader):
    model.eval()
    y_scores=[]
    y_real=[]
    acc_score =[]


    for batch_id, [image_1,image_2,is_match] in enumerate(data_loader):


        image_1, image_2 = image_1.to(device), image_2.to(device)

        matches_at_threshold = predict_score(model,image_1, image_2)

        for i in is_match:
            y_real.append(i)

        for i in matches_at_threshold:
            y_scores.append(i)
        #print("Shape of y scores:",len(y_scores))
        #print("Shape of y real :",len(y_real))
        
    return y_scores, y_real 


def validation(epoch,valid_loader,optimizer,criterion) :
    model.eval() 
    print("In evaluation -->")
    total_accuracy=0
    total=0
    valid_loss=0
    for batch_idx,(x,y) in enumerate(valid_loader):
        accuracy=0
        x=x.to(device)
        y = y.to(device)
        
        output=model(x)
        loss=criterion(output,y)
        valid_loss+=loss.item()
        #We have to calculate accuracy
        predictions=F.softmax(output,dim=1)
        _,top_prediction=torch.max(predictions,1)#To get the 
        
        top_pred_labels=top_prediction.view(-1)
        accuracy+=torch.sum(torch.eq(top_pred_labels,y)).item()
        total+=len(y)
        total_accuracy+=accuracy
        #print('E: %d, B: %d / %d, Valid Loss: %.3f | avg_loss: %.3f,  Validation Accuracy : %.4f' % (epoch+1, batch_idx+1, len(valid_loader),loss.item(),valid_loss/(batch_idx+1),(accuracy*100)/len(y)),end='\n ')
        del x
        del y
    model.train()    
    print(" Validation Accuracy is {} at Epoch {} ".format((total_accuracy*100)/total ,epoch+1))
    return (total_accuracy*100)/total
    
#Training loop
def train(num_epochs,train_loader,valid_loader,optimizer,criterion,valid_ver_dataloader):
    print('Num epochs:',num_epochs)
    
    val_acc=[]
    for epoch in range(num_epochs):
        model.train()
        print(' epoch :',epoch+1)
        train_loss=0
 
        for batch_idx,(x,y) in enumerate(train_loader):
            start=time.time()
            x=x.to(device)
            y = y.to(device)

            output=model(x)
            #print(output.shape)
            loss=criterion(output,y)

            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()

            train_loss+=loss.item()
            stop=time.time()
            #print('E: %d, B: %d / %d, Train Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f' % (epoch+1, batch_idx+1, len(train_loader),loss.item(),train_loss/(batch_idx+1),stop-start),end='\n ')
    
            #torch.cuda.empty_cache()        
            del loss
            del y
            del x
        val_accu=validation(epoch,valid_loader,optimizer,criterion)

        if(epoch==num_epochs-1):
            torch.save(model.state_dict(), "hw2p2_checkpoint_Resnet34_exp2.pth")    
            
        val_acc.append(val_accu)
        #AUC Score generator for Valid_Set
        y_scores, y_real=score_generator(valid_ver_dataloader)
        print("AUC of ROC at Epoch {} is :{} ".format(epoch+1,sklearn.metrics.roc_auc_score(y_real, y_scores)))
        
        if sklearn.metrics.roc_auc_score(y_real, y_scores)>=92.0:
            print("This was the best ROC , now we stop!!!")
            break
        
    return val_acc


#Exp2--> two with SGD, learning rate=0.15,wieght decay=5e-5,batch size=200 for training
#Set the hyperparameters here
num_epochs=10
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),momentum=0.9,lr=0.15,weight_decay=5e-5)
val_acc=train(num_epochs,train_loader,valid_loader,optimizer,criterion,valid_ver_dataloader)  


#Now for Testing
class Custom_dataset_test():
    def __init__(self,path,transform):
        self.img1=path[:,0]
        self.img2=path[:,1]
        self.transform=transform
             
    def __len__(self):
        return self.img1.shape[0]
    
    def __getitem__(self,index): 
        
        dir='/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2'
        img1=Image.open(os.path.join(dir,self.img1[index]))
        img2=Image.open(os.path.join(dir,self.img2[index]))
        img1=self.transform(img1)
        img2=self.transform(img2)
        
        return img1,img2

test_set=Custom_dataset_test(test_list,transform)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle = False, num_workers=16) 

#We generate scores for the test_set
def score_generator_test(data_loader):
    y_scores=[]
   
    acc_score =[]

    for batch_id, [image_1,image_2] in enumerate(data_loader):

        image_1, image_2 = image_1.to(device), image_2.to(device)

        matches_at_threshold = predict_score(model,image_1, image_2)

        for i in matches_at_threshold:
            y_scores.append(i)
        print("Shape of y scores:",len(y_scores))
                
    return y_scores 

y_test_scores=score_generator_test(test_dataloader)

for i in range(len(y_test_scores)):
    y_test_scores[i]=y_test_scores[i].numpy()

sample_csv=pd.read_csv('/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/hw2p2_sample_submission.csv')

#We read in the sample csv and replace its last column with the scores we have achieved
n = sample_csv.columns[1]
sample_csv[n] = y_test_scores

submission_csv=sample_csv

submission_csv.to_csv('/home/ironman/shayeree/records/11-785-fall-20-homework-2-part-2/my_submission_hw2p2_2_Resnet_34_2.csv',index=False) 
