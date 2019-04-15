## OptimizeMVS

Created by [Yi Wei](https://github.com/weiyithu), [Shaohui Liu](http://github.com/B1ueber2y/) and [Wang Zhao](https://github.com/thuzhaowang) from Tsinghua University.

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


### Acknowledgements
Part of the external operators are borrowed from [latent_3d_points](https://github.com/optas/latent_3d_points) and [PointNet++](https://github.com/charlesq34/pointnet2). The multi-view images were rendered from [ShapeNetCore](https://www.shapenet.org/) with the preprocessing scripts in [mvcSnP](https://github.com/shubhtuls/mvcSnP) and the point cloud data was from [latent_3d_points](https://github.com/optas/latent_3d_points).

This work was supported in part by the National Natural Science Foundation of China under Grant U1813218, Grant 61822603, Grant U1713214, Grant 61672306, and Grant 61572271.


### Citation
If you find this work useful in your research, please consider citing:

    @inproceedings{wei2019conditional,
      author = {Wei, Yi and Liu, Shaohui and Zhao, Wang and Lu, Jiwen and Zhou, Jie},
      title = {Conditional Single-view Shape Generation for Multi-view Stereo Reconstruction},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2019}
    }
The first three authors share equal contributions.
