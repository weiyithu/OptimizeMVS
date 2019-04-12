## OptimizeMVS

Created by [Wei Yi](https://github.com/weiyithu), [Shaohui Liu](http://b1ueber2y.me/), [Wang Zhao](https://github.com/thuzhaowang), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jie Zhou](https://www.tsinghua.edu.cn/publish/auen/1713/2011/20110506105532098625469/20110506105532098625469_.html) from Tsinghua University.

### Introduction
This repository is for the paper [Conditional Single-view Shape Generation for Multi-view Stereo Reconstruction](https://arxiv.org/abs/1612.01105), which is going to appear in CVPR 2019. 

![prediction example](https://github.com/weiyithu/OptimizeMVS/doc/teaser.png)

### Installation
The code has been tested with Python 2.7, tensorflow 1.3.0 on Ubuntu 16.04.

#### 1. Clone code
```bash
git clone https://github.com/weiyithu/OptimizeMVS.git
```

#### 2. Install packages

It's recommended to use Python virtual environment.
```bash
cd OptimizeMVS
virtualenv env
source ./env/bin/activate
pip install -r requirements.txt
```


### Usage


### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{zhao2017pspnet,
      author = {Hengshuang Zhao and
                Jianping Shi and
                Xiaojuan Qi and
                Xiaogang Wang and
                Jiaya Jia},
      title = {Pyramid Scene Parsing Network},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
    }
