# HW3

## Table of Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [HW3](#hw3)
	* [Table of Content](#table-of-content)
	* [Requirements](#requirements)
	* [Usage](#usage)
	* [Results](#results)
		

<!-- /code_chunk_output -->

## Requirements
  * Python 3.6
  * Tensorflow 1.6
  * keras 2.0.7
  * scikit-learn 0.19.1
  * h5py 2.7.1
  * matplotlib
  * numpy
  * scipy

## Usage

  * **Training**

    **Choose one argument in [...] list**
    ```
    python3 train.py --batch-size 8 --epochs 20 --arch [FCN32s, FCN16s, FCN8s]
    ```

  * **Visualization of Validation**

    ```
    python3 validation.py
    ```

    Generates a file "results", you may check the figure generated.
    ```
    cd results;
    ```

  * **Calculate Mean IOU**

    ```
    python3 mean_iou_evaluate.py -g [ground truth directory] -p [predict directory]
    ```

## Results

  	**VGG16-FCN32s**

    class     | accuracy  |
    --------- | ----------
    class 0   | 0.74253
    class 1   | 0.86813
    class 2   | 0.30609
    class 3   | 0.77914
    class 4   | 0.72905
    class 5   | 0.61108

    mean iou score: 0.672671

    **VGG16-FCN16s**

    class     | accuracy  |
    --------- | ----------
    class 0   | 0.74944
    class 1   | 0.87892
    class 2   | 0.28751
    class 3   | 0.77799
    class 4   | 0.72519
    class 5   | 0.65657

    mean iou score: 0.679271

    **VGG16-FCN8s**

    class     | accuracy  |
    --------- | ----------
    class 0   | 0.76292
    class 1   | 0.87474
    class 2   | 0.26423
    class 3   | 0.73023
    class 4   | 0.71806
    class 5   | 0.61627

    mean iou score: 0.661078
    
