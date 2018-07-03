# Final Project

<!-- /code_chunk_output -->

## Tasks
  * Task 1: Small Data Supervised Learning
    * Train classifier using a limited version of Fashion MNIST dataset 
    * Apply transfer learning
  * Task 2: One-shot / Few-shot Learning (Not done, performance not good)
    * Design model to recognize a number of novel classes with insufficient number of training images

## Requirements
  * Python 3.6.4
  * Torch 0.4.0
  * torchvision 0.2.0
  * scipy 1.0.1
  * matplotlib 2.2.2
  * numpy 1.14.2

## Dataset
   * For task 1, FashionMNIST with selected data
       * 2000 training samples with labels 
       * 10000 testing samples with no labels
   * For task 2, Cifar-100 
       * 80 base classes with 500/100 training/testing samples
       * 20 novel classes with 1/5/10 training samples


## Implementation
   * **Small Data Supervised Learning**
  
      First, get into the root file for task 1, `cd task1`
      
      Put the file `Fashion_MNIST_student` into the directory `datasets` as shown
      
      * For training, please check the argument listed in `train.py`
        
        ```
            python3 train.py --train-dir datasets/Fashion_MNIST_student/train
                             --test-dir datasets/Fashion_MNIST_student/test
                             --batch-size 128
                             --epochs 100
                             --save-freq 1
        ```       
           
      * For testing, please check the argument listed in `inference.py`
      
        ```
            python3 train.py --train-dir datasets/Fashion_MNIST_student/train
                             --test-dir datasets/Fashion_MNIST_student/test
                             --checkpoint checkpoints/fashion_mnist/best_checkpoint.pth.tar
        ``` 
      
      * For Kaggle results, execute the script `task1.sh` for download model and inference the result
        
        ```
            bash task1.sh
        ```
        
        After execution, there will be a file called `result.csv`
        
   * **One-shot / Few-shot Learning**
      
      Alternative implementation of **Learning to Compare: Relation Network for Few-Shot Learning** in PyTorch 0.4.0
      
      Reference: **Learning to Compare: Relation Network for Few-Shot Learning**, Sung et al., CVPR'18 ([paper](https://arxiv.org/pdf/1711.06025.pdf), [code](https://github.com/floodsung/LearningToCompare_FSL))
   
      * **Usage**
        
        `cd task2` first, check configuration in `config/relationnet_config.json` before training
        
        * Training
            ```
                python3 train.py --config config/relationnet_config.json
            ```

      * **Citation**
        
        ```
            @inproceedings{sung2018learning,
              title={Learning to Compare: Relation Network for Few-Shot Learning},
              author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
              year={2018}
             }
        ```
