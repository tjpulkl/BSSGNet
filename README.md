This repo is a PyTorch implementation of our paper "Channel and Spatial Enhancement Network for Human Parsing" accepted by lmage and Vision Computing. 
 
The dominant backbones of neural networks for scene parsing consist of multiple stages, where feature maps in different stages often contain varying levels of spatial and semantic information. High-level features convey more semantics and fewer spatial details, while low-level features possess fewer
semantics and more spatial details. Consequently, there are semantic-spatial gaps among features at different levels, particularly in human parsing tasks. Many existing approaches directly upsample multi-stage features and aggregate them through addition or concatenation, without addressing the
semantic-spatial gaps present among these features. This inevitably leads to spatial misalignment, semantic mismatch, and ultimately misclassification in parsing, especially for human parsing that demands more semantic information and more fine details of feature maps for the reason of
intricate textures, diverse clothing styles, and heavy scale variability across different human parts. In this paper, we effectively alleviate the long-standing challenge of addressing semantic-spatial gaps between features from different stages by innovatively utilizing the subtraction and addition
operations to recognize the semantic and spatial differences and compensate for them. Based on these principles, we propose the Channel and Spatial Enhancement Network (CSENet) for parsing, offering a straightforward and intuitive solution for addressing semantic-spatial gaps via injecting
high-semantic information to lower-stage features and vice versa, introducing fine details to higher-stage features. Extensive experiments on three dense prediction tasks have demonstrated the efficacy of our method. Specifically, our method achieves the best performance on the LIP and CIHP datasets
and we also verify the generality of our method on the ADE20K dataset.

Requirements Pytorch 1.9.0

Python 3.7

Implementation
Dataset
Please download LIP dataset and make them follow this structure:
'''
|-- LIP
    |-- images_labels
        |-- train_images
        |-- train_segmentations
        |-- val_images
        |-- val_segmentations
        |-- train_id.txt
        |-- val_id.txt
'''
Please download imagenet pretrained resent-101 from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into dataset folder.

Training and Evaluation
./run.sh
Please download the trained model for LIP dataset from [baidu drive](https://pan.baidu.com/s/1-9pR_ycvqkWBnDoKyI2nUw?pwd=bb9c) and put it into snapshots folder.

run ./run_evaluate_multiScale.sh for multiple scale evaluation.

The parsing result of the provided 'LIP_epoch_Best.pth' is 61.27 on LIP dataset.

Citation:

If you find our work useful for your research, please cite:

@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Kunliang and Choi, Ouk and Wang, Jianming and Hwang, Wonjun},
    title     = {CDGNet: Class Distribution Guided Network for Human Parsing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4473-4482}
}



@article{SSUCSFR2024,
    title={ Channel and Spatial Enhancement Network for Human Parsing },
    journal = {Image and Vision Computing (\textit{IVC})},
    author={Kunliang Liu, Rize Jin, Yuelong Li, Jianming Wang, and Wonjun Hwang },   
    year = {2024}
}

............

Acknowledgement:

  We acknowledge Ziwei Zhang and Tao Ruan for sharing their codes.
