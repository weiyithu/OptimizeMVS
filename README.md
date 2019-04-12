## OptimizeMVS

Created by [Yi Wei](https://github.com/weiyithu), [Shaohui Liu](http://b1ueber2y.me/) and [Wang Zhao](https://github.com/thuzhaowang) from Tsinghua University.

### Introduction
This repository contains source code for [Conditional Single-view Shape Generation for Multi-view Stereo Reconstruction](https://arxiv.org/abs/1612.01105) in tensorflow. 

![prediction example](https://github.com/weiyithu/OptimizeMVS/blob/master/doc/teaser.png)

### Installation
The code has been tested with Python 2.7, tensorflow 1.3.0 on Ubuntu 16.04.

#### 1. Clone code
```bash
git clone https://github.com/weiyithu/OptimizeMVS.git
```

#### 2. Install packages

Python virtual environment is recommended.
```
cd OptimizeMVS
virtualenv env
source ./env/bin/activate
pip install -r requirements.txt
```


### Usage


### Citation
If you find this work useful in your research, please consider citing:

    @inproceedings{wei2019conditional,
      author = {Wei, Yi and Liu, Shaohui and Zhao, Wang and Lu, Jiwen and Zhou, Jie},
      title = {Conditional Single-view Shape Generation for Multi-view Stereo Reconstruction},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2019}
    }
The first three authors share equal contributions.
