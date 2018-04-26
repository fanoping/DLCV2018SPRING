# HW3

## Table of Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [HW3](#hw3)
	* [Table of Content](#table-of-content)
	* [Requirements](#requirements)
	* [Instuctions](#instuctions)
		* [1-1-1 Simulate a function](#1-1-1-simulate-a-function)
		* [1-1-2 Train on actual tasks](#1-1-2-train-on-actual-tasks)
	* [1-2 Optimization](#1-2-optimization)
		* [1-2-1 Visualize the optimization process](#1-2-1-visualize-the-optimization-process)
		* [1-2-2 Observe gradient norm during training](#1-2-2-observe-gradient-norm-during-training)
		* [1-2-3 What happens when gradient is almost zero](#1-2-3-what-happens-when-gradient-is-almost-zero)
		* [1-2-B Bonus: Error surface](#1-2-b-bonus-error-surface)

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

## Instructions

### 1-1-1 Simulate a function
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

  * **Results**

    First Header | Second Header
    ------------ | -------------
    Content from cell 1 | Content from cell 2
    Content in the first column | Content in the second column
