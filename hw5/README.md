# HW5

<!-- /code_chunk_output -->

## Task
  * Feature extraction from pre-trained CNN models
    * VGG19 
    * Resnet50 (Done)
    * Densenet121
  * Trimmed action recognition
    * Training on RNN with sequences of CNN features and labels
  * Temporal action segmentation
    * Extend RNN model for sequence-to-sequence prediction

## Requirements
  * Python 3.6.4
  * Tensorflow 1.6
  * Torch 0.4.0
  * torchvision 0.2.0
  * sklearn 0.19.1
  * skimage 0.13.1
  * skvideo 1.1.10
  * scipy 1.0.1
  * matplotlib 2.2.2
  * numpy 1.14.2

## Dataset
   * 29 Videos with frame size 240 * 320, 11 categories for label
   * For task 1 & 2, videos are trimmed into frames (3236 training samples / 517 validation samples)
   * For task 3, full video length is required (29 training samples / 5 validation samples)

## Data Pre-process
   
   * **Train**
     
     ```
        python3 preprocess.py --mode [trimmed, full-length]
                              --csv-file HW5_data/TrimmedVideos/label/gt_train.csv 
                              --video-dir HW5_data/TrimmedVideos/video/train 
                              --video-file train_video.tar
                              --full-length-dir HW5_data/FullLengthVideos/videos/train
                              --full-length-file train_full_length_video.tar
     ```
   * **Validation**
     
     ```
        python3 preprocess.py --mode [trimmed, full-length]
                              --csv-file HW5_data/TrimmedVideos/label/gt_valid.csv 
                              --video-dir HW5_data/TrimmedVideos/video/valid 
                              --video-file valid_video.tar
                              --full-length-dir HW5_data/FullLengthVideos/videos/valid
                              --full-length-file valid_full_length_video.tar
     ```

## Implementation
   * **Feature extraction from pre-trained CNN models**
  
      * **Usage**
            
        * Training
            
            ```
            python3 train.py --arch cnn --pretrained Resnet50 
                             --epochs 150 --batch-size 128 --save-freq 1 --verbosity 1      
            ```
            
        * Visualize / Inference
        
            ```
            python3 cnn_inference.py --input-feature cnn_valid_feature.tar 
                                     --input-video valid_video.tar
                                     --input-csv HW5_data/TrimmedVideos/label/gt_valid.csv
                                     --video-dir HW5_data/TrimmedVideos/video/valid
                                     --output-file saved/cnn
                                     --checkpoint checkpoints/cnn_resnet50/epoch300_checkpoint.pth.tar
                                     --pretrained Resnet50
            ```
      * **Results**
        
        train     | valid     |
        --------- | ----------
        0.993823  | 0.485493   
  
  * **Trimmed action recognition**
    
    * **Usage**
    
        * Training
            
            ```
            python3 train.py --arch rnn --pretrained Resnet50 
                             --epochs 150 --batch-size 128 --save-freq 1 --verbosity 1      
            ```
            
        * Visualize / Inference
        
            ```
            python3 rnn_inference.py --input-feature cnn_valid_feature.tar 
                                     --input-video valid_video.tar
                                     --input-csv HW5_data/TrimmedVideos/label/gt_valid.csv
                                     --video-dir HW5_data/TrimmedVideos/video/valid
                                     --output-file saved/rnn
                                     --checkpoint checkpoints/rnn_resnet50/epoch150_checkpoint.pth.tar
                                     --pretrained Resnet50
            ```
    * **Results** 
    
        train     | valid     |
        --------- | ----------
        0.841980  | 0.520309  
        
  *	**Temporal action segmentation**
        
     * **Usage**
    
        * Training
            
            ```
            python3 train.py --arch seq2seq --pretrained Resnet50 
                             --epochs 300 --batch-size 8 --save-freq 1 --verbosity 1      
            ```
            
        * Visualize / Inference
        
            ```
            python3 seq2seq_inference.py --input-feature rnn_full_length_valid_feature.tar
                                         --full-length-dir HW5_data/FullLengthVideos/videos/valid
                                         --full-length-file valid_full_length_video.tar
                                         --output-file saved/seq2seq
                                         --checkpoint checkpoints/seq2seq_resnet50/epoch150_checkpoint.pth.tar
                                         --pretrained Resnet50
            ```
     * **Results**
        
        train     | valid     |
        --------- | ----------
        0.752159  | 0.566797
     
## Learning curve / Results
   * See figures in the directory "saved"
       ```
       cd saved; cd [cnn/rnn/seq2seq]
       ```
    
    
