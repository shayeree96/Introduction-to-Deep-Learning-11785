
1. The ‘submission.csv’ file gets generated at the end of the training, on the test dataset
2. Remember to change the path names for the trainset, validset, testset, train_labels and valid_labels 
3. Check point after the reaching the required the Levenshtein distance between 6-7 is generated with filename  'Model_final_kaggle.pth’ 

Hyperparameters Used:

1. Activation Function :ReLU
2. Optimizer : Adam with learning rate=1e-3 , momentum=0.9, weight decay rate=5e-5
3. Loss Function : CTC Loss
4. Scheduler : Step Scheduler was used with step size=1 and gamma=0.85
5. Beam Width in CTC Decoder : 20

Please note the following for Training and Validation:
1. No of epochs : 15-17 epochs (based on the Levenshtein distance on the Validation Dataset)
Please note the following for testing and validation verification lists:
1. Levenshtein distance has been used as the metric for counting how many additions, deletions and modifications are required to make one sequence into another
 Model Architecture :

Model((convd1): Conv1d(13, 512, kernel_size=(3,), stride=(1,), padding=(1,))
    (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (convd2): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
    (bn2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (lstm): LSTM(512, 512, num_layers=4, bidirectional=True)
    (l1): Linear(in_features=1024, out_features=256, bias=True)
    (l2): Linear(in_features=256, out_features=42, bias=True))

Data Loading Scheme :

1. Batch Size in train dataloader : 64
2. Batch Size for validation dataloader : 32
3. The CTC Beam Decoder with the Beam Width of 20 is used to search for the best output sequence and the PHONEME MAP is used to map the indexes found to the respective phonemes

Levenshtein distance Score wrt to Test Set on Kaggle : 7.300
