# HW5

<!-- /code_chunk_output -->

## Task
  * Feature Extraction from pre-trained CNN models
    * VGG19
    * Resnet50
    * Densenet121
  * Trimmed Action Recognition
    * Training on RNN with sequences of CNN features and labels
  * Temporal Action Segmentation
    * Extend RNN model for sequence-to-sequence prediction

## Requirements
  * Python 3.6
  * Tensorflow 1.6
  * Torch 0.4.0
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

## Preprocess
   
   * **Train**
     
     ```
        python3 preprocess.py --csv-file HW5_data/TrimmedVideos/label/gt_train.csv 
                              --video-dir HW5_data/TrimmedVideos/video/train 
                              --video-file train_video.tar
                              --full-length-dir HW5_data/FullLengthVideos/videos/train
                              --full-length-file train_full_length_video.tar
     ```
   * **Valid**
     
     ```
        python3 preprocess.py --csv-file HW5_data/TrimmedVideos/label/gt_valid.csv 
                              --video-dir HW5_data/TrimmedVideos/video/valid 
                              --video-file valid_video.tar
                              --full-length-dir HW5_data/FullLengthVideos/videos/valid
                              --full-length-file valid_full_length_video.tar
     ```

## Implementation
   * **Feature Extraction from pre-trained CNN models**
  
      * **Usage**
            
        * Training
            
            ```
            python3 train.py --arch cnn --pretrained Resnet50 
                             --epochs 150 --batch-size 128 --save-freq 1 --verbosity 1      
            ```
            
        * Visualize/ Inference
        
            ```
            python3 cnn_inference.py --input-feature cnn_valid_feature.tar 
                                     --input-csv HW5_data/TrimmedVideos/label/gt_valid.csv
                                     --output-file saved/cnn
                                     --checkpoint checkpoints/cnn_resnet50/epoch150_checkpoint.pth.tar
                                     --pretrained Resnet50
            ```
      * **Results**
  
  * **Trimmed Action Recognition**
    
    * **Usage**
    
        * Training
            
            ```
            python3 train.py --arch rnn --pretrained Resnet50 
                             --epochs 150 --batch-size 128 --save-freq 1 --verbosity 1      
            ```
            
        * Visualize/ Inference
        
            ```
            python3 rnn_inference.py --input-feature cnn_valid_feature.tar 
                                     --input-csv HW5_data/TrimmedVideos/label/gt_valid.csv
                                     --output-file saved/rnn
                                     --checkpoint checkpoints/rnn_resnet50/epoch150_checkpoint.pth.tar
                                     --pretrained Resnet50
            ```
    * **Results**     
  *	**Temporal Action Segmentation**
        
     * **Usage**
    
        * Training
            
            ```
            python3 train.py --arch seq2seq --pretrained Resnet50 
                             --epochs 300 --batch-size 8 --save-freq 1 --verbosity 1      
            ```
            
        * Visualize/ Inference
        
            ```
            python3 seq2seq_inference.py --input-feature rnn_full_length_valid_feature.tar
                                         --output-file saved/seq2seq
                                         --checkpoint checkpoints/seq2seq_resnet50/epoch150_checkpoint.pth.tar
                                         --pretrained Resnet50
            ```
     * **Results**
     
## Learning curve / Results
   * See figures in the directory "saved"
       ```
       cd saved; cd [cnn/rnn/seq2seq]
       ```
    
    
