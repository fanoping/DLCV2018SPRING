# HW4

<!-- /code_chunk_output -->

## Task
  * Image Generation
    * Variational Autoencoder (VAE)
    * Generative Adversarial Network (GAN)
  * Feature Disentanglement
    * Auxiliary Classifier Generative Adversarial Network (AC-GAN)
    * Information Maximizing Generative Adversarial Network (Info-GAN) 

## Requirements
  * Python 3.6
  * Tensorflow 1.6
  * Torch 0.3.1
  * Numpy 1.14.2
  * Pandas 0.22.0
  * sckit-learn 0.19.1
  * scipy 1.0.1
  * Numpy 1.14.2

## Dataset
   * CelebFaces Attribute(CelebA) Dataset (40000 training samples / 2621 testing samples)
   * Images are cropped and downscaled to 64x64 
   * 13 out of 40 attributes are provided for experiment

## Image Generation
  * **Variational Autoencoder**
    
    * **Usage**
    
        * Training
            
            ```
            python3 train.py --arch vae --epochs 300 --batch-size 128 --save-freq 1
            ```
            
        * Visualize/ Inference
        
            ```
            python3 vae_inference.py --input-file hw4_data/test --output-file saved/vae \
                                        --checkpoint checkpoints/vae/best_checkpoint.pth.tar
            ```
  
  * **Generative Adversarial Network (GAN)**
    
    * **Usage**
    
        * Training
    
            ```
            python3 train.py --arch gan --epochs 300 --batch-size 128 --save-freq 1
            ```
        
        * Visualize/ Inference
            
            choose the training epoch of the checkpoint
            ```
            python3 gan_inference.py --output-file saved/gan \
                                     --checkpoint checkpoints/gan/epoch231_checkpoint.pth.tar
            ```
    

## Feature Disentanglement

  *	**Auxiliary Classifier Generative Adversarial Network (AC-GAN)**
    
        * **Usage**
    
            * Training
    
                ```
                python3 train.py --arch acgan --epochs 300 --batch-size 128 --save-freq 1
                ```
        
            * Visualize/ Inference
            
                choose the training epoch of the checkpoint
                ```
                python3 acgan_inference.py --output-file saved/acgan \
                                     --checkpoint checkpoints/acgan/epoch300_checkpoint.pth.tar
                ```

  * **Information Maximizing Generative Adversarial Network (Info-GAN)**
    
      * **Usage**
    
          * Training
    
            ```
            python3 train.py --arch infogan --epochs 300 --batch-size 128 --save-freq 1
            ```
        
          * Visualize/ Inference
            
            choose the training epoch of the checkpoint
            ```
            python3 infogan_inference.py --output-file saved/infogan \
                                     --checkpoint checkpoints/infogan/epoch300_checkpoint.pth.tar
            ```

## Results
   * See figures in the directory "saved"
       ```
       cd saved; cd [vae/gan/acgan/infogan]
       ```
    
    
