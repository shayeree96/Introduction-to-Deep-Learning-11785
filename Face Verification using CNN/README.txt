1.Run the model using command python hw2p2_Submission.py
2.The ‘my_submission_hw2p2_2_Resnet_34.csv’ file gets generated at the end of the training on the test verification dataset
3.Remember to change the path names for the train_dir, valid_dir, test_list and valid_list 
4.Check point after the reaching the required AUC gets generated with filename  ‘hw2p2_checkpoint_Resnet34_exp2.pth’ 
(The checkpoint can be loaded by uncommenting lines 131 and 133 )
5.Kindly remember to change the cuda device id in line 132 in nn.DataParallel

Hyperparameters Used:

1. Activation Function :ReLU
2. Optimizer : SGD with learning rate=0.15 , momentum=0.9, weight decay rate=5e-5
3. Loss Function : Cross Entropy Loss

Please note the following for Training and Validation:
1. No of epochs : 10 - 15 ( epochs can be lesser if AUC of 92 and above is reached while training )
Please note the following for testing and validation verification lists:
1. Cosine Similarity has been used as the metric for calculating the similarity scores in the predict_score function
 Model Architecture :

Resnet 34 using Basic Block class and layers [ 3,4,6,3 ]

Data Loading Scheme :

1. Batch Size in train dataloader : 128
2. Batch Size for validation dataloader : 64
3. In the __init__() of  MyDataset class, we store all the images belonging to a Face ID  in X and its corresponding label as Face_ID in Y. In the __getitem__() function we open the image and transform it and return the image and its corresponding Face_ID label
4. In the __init__() of  Custom_Dataset class , we split the validation verification list, where img1  and img2 save the pairwise images and Y stores 1 or 0 depending on whether the images are a match or not respectively.  In the __getitem__() function we open the images and transform them and return the images and its corresponding label of whether they are a match or not
5.4. In the __init__() of  Custom_Dataset_test class , we split the test verification list, where img1  and img2 save the pairwise images  In the __getitem__() function we open the images and transform them and return the images

Note: Validation Accuracy after 10 epochs : 70.88 %
AUC score  after 10 epochs : 0.9255
