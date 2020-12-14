
1. The ‘submission.csv’ file gets generated at the end of the training, on the test dataset
2. Remember to change the path names for the speech_train, speech_valid, speech_test, transcript_train and transcript_valid

Hyperparameters Used:

1. Activation Function :SELU
2. Optimizer : Adam with learning rate=1e-3 
3. Loss Function : Cross Entropy Loss (Masked)
4. Teacher Forcing Rate varying across the epochs, starting from 0.1 and reducing by 0.05 every 10 epochs

Regularizations Used:

1. Locked Dropout in the pBLSTM layers
2. Weight Dropout in the pBLSTM layers
3. Weight Tying for the character embedding layer and the character probability layer in the decoder

Weight Initialization: From Uniform Distribution b/w [-0.1,0.1] for the character embedding layer and the character probability layer

Please note the following for Training and Validation:
1. No of epochs : 35 epochs (based on the Levenshtein distance on the Validation Dataset)
 Model Architecture :

Seq2Seq(
  (encoder): Encoder(
    (lstm): LSTM(40, 256, batch_first=True, bidirectional=True)
    (pBLSTM1): pBLSTM(
      (blstm): LSTM(1024, 256, batch_first=True, dropout=0.2, bidirectional=True)
      (lockdrop): LockedDropout()
    )
    (pBLSTM2): pBLSTM(
      (blstm): LSTM(1024, 256, batch_first=True, dropout=0.2, bidirectional=True)
      (lockdrop): LockedDropout()
    )
    (pBLSTM3): pBLSTM(
      (blstm): LSTM(1024, 256, batch_first=True, dropout=0.2, bidirectional=True)
      (lockdrop): LockedDropout()
    )
    (key_network): Linear(in_features=512, out_features=256, bias=True)
    (value_network): Linear(in_features=512, out_features=256, bias=True)
    (lockdrop): LockedDropout()
  )
  (decoder): Decoder(
    (embedding): Embedding(35, 512, padding_idx=0)
    (lstm1): LSTMCell(768, 256)
    (lstm2): LSTMCell(256, 256)
    (attention): Attention()
    (character_prob): Linear(in_features=512, out_features=35, bias=True)
    (fc1): Linear(in_features=256, out_features=256, bias=True)
    (selu1): SELU()
    (selu2): SELU()
    (fc2): Linear(in_features=256, out_features=256, bias=True)
  )
)

Data Loading Scheme :

1. Batch Size in train dataloader : 128
2. Batch Size for validation dataloader : 64
3. Greedy Search has been used for decoding the predictions for validation and test set and also for predictions in the decoder. The hidden dimensions for the encoder and the decoder have been increased. 
4. The context from attention (masked) and the output from the LSTM-2 cell of the decoder are each passed into a linear layer has been used followed by an activation layer. Thereafter they are concatenated to find the final prediction
5. During training if variable drawn from random.random is greater than the Teacher Forcing Rate, then we will use the groundtruth from the previous time step, otherwise just the argmax of the previous time step.
6. During testing we will simply use greedy search for feeding into the current time step from previous time step.
7. For both training and testing we will feed into the embedding of <sos> at i=0 in the loop
8. We use the class Speech2TextDataset to transform the transcript_train and transcript_valid into indices using the letter to index dictionary and append the index of <eos> to the end of every sentence.

Levenshtein distance Score wrt to Test Set on Kaggle : 18.32
