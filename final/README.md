# Final Project

<!-- /code_chunk_output -->

## Tasks
  * Task 1: Small Data Supervised Learning
    * Train classifier using a limited version of Fashion MNIST dataset 
    * Apply transfer learning
  * Task 2: One-shot / Few-shot Learning
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
  
      * **Usage**
            
        * Training
            
        * Visualize / Inference
        
   * **One-shot / Few-shot Learning**
      
      Alternative implementation of **Prototypical Networks for Few Shot Learning** in PyTorch 0.4.0
      
      Reference: **Prototypical Networks for Few-shot Learning**, Snell et al., NIPS'17 ([paper](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf), [code](https://github.com/jakesnell/prototypical-networks))
      
      * **Usage**
        
        `cd task2` first, check configuration in `config/protonet_config.json` before training
        
        * Training
            ```
                python3 train.py --config config/protonet_config.json
            ```
            
        * Visualize / Inference

      * **Citation**
        
        ```
            @inproceedings{snell2017prototypical,
              title={Prototypical Networks for Few-shot Learning},
              author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
              booktitle={Advances in Neural Information Processing Systems},
              year={2017}
             }
        ```
      
      Alternative implementation of **Learning to Compare: Relation Network for Few-Shot Learning** in PyTorch 0.4.0
      
      Reference: **Learning to Compare: Relation Network for Few-Shot Learning**, Sung et al., CVPR'18 ([paper](https://arxiv.org/pdf/1711.06025.pdf), [code](https://github.com/floodsung/LearningToCompare_FSL))
   
      * **Usage**
        
        `cd task2` first, check configuration in `config/relationnet_config.json` before training
        
        * Training
            ```
                python3 train.py --config config/relationnet_config.json
            ```
            
        * Visualize / Inference

      * **Citation**
        
        ```
            @inproceedings{sung2018learning,
              title={Learning to Compare: Relation Network for Few-Shot Learning},
              author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
              year={2018}
             }
        ```
