1.Run the model using command python hw1p2_Submission.py
2.The .csv file gets generated at the end of the training and testing 
3.Remember to change the path names for the trainset, valid, testset and train_labels and valid_labels
4.Check point after the last epoch gets generated with filename  model_gelu.pth
5.Kindly remember to change the cuda device id in line 95 in nn.DataParallel

Hyperparameters Used:
1. Context Size=22
2. Activation Function :GeLU
3. Optimizer : Adam with learning rate=1e^-3 and default weight decay rate=1e^-5
4. Loss Function : Cross Entropy Loss

Please note the following for training and Validation:
1. No of epochs :2
Please note the following for testing:
1. Batch Size for testing: 4096
 Model Architecture :

1. Layer1: Linear Layer with input size (13*(2*context+1),1024) followed by 1D BatchNorm(1024) followed by GELU() activation function
2. Layer2: Linear Layer with input size (1024,2048) followed by 1D BatchNorm(2048) followed by GELU() activation function
3. Layer3: Linear Layer with input size (2048,4096) followed by 1D BatchNorm(4096) followed by GELU() activation function
4. Layer4: Linear Layer with input size (4096,2048) followed by 1D BatchNorm(2048) followed by GELU() activation function
5. Layer5: Linear Layer with input size (2048,1024) followed by 1D BatchNorm(1024) followed by GELU() activation function
6. Layer6: Linear Layer with input size (1024,512) followed by 1D BatchNorm(512) followed by GELU() activation function
7. Layer7: Linear Layer with input size (512,346) followed by Softmax 

Data Loading Scheme :

1. Batch Size in train dataloader : 4096
2. Batch Size for validation dataloader : 2048
3. Used a context Size of 22 for each phoneme X resulting in (2*22+1),13 (45,13) size for each input X and (13,) for label Y, and this was implemented in __getitem__ method in MyDataset Class
4. Padded each of the rows of in the 1D array for X, here 22,002 rows with zeros above and below depending the context size, in my case 22 in __init__() in MyDataset Class
5. Made a list of tuples in __init__() to save the possible combination of indexes in self.index_ij for grouping the context sizes in get_item method  in MyDataset Class, when the model is iterating over the dataloader

Note: Validation Accuracy after 2 epochs : 73.88 %
