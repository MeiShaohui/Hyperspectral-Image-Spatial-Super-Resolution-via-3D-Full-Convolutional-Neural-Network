# Hyperspectral Image Spatial Super-Resolution via 3D Full Convolutional Neural Network

By [Shaohui Mei](http://teacher.nwpu.edu.cn/en/meishaohui.html), Xin Yuan, [Jingyu Ji](https://github.com/JiJingYu) , [Yifan Zhang](http://teacher.nwpu.edu.cn/en/zhangyifan.html), [Shuai Wan](http://teacher.nwpu.edu.cn/en/wanshuai.html), [Qian Du](http://my.ece.msstate.edu/faculty/du/) 

---
### Introduction

The 3D-FRCNN is an unified framework for hyperspectral image super-resolutiom(SR) with a single network. You can use the code to train/evaluate a network for hsi super-resolution(SR). For more details, please refer to our [paper](http://www.mdpi.com/2072-4292/9/11/1139/htm).

### Proposed Framework
![](./figure/Proposed_Framework.png) 

### Some SR results 
![](./figure/Some_SR_results.png) 

---
### Citing our work
```
@Article{rs9111139,
AUTHOR = {Mei, Shaohui and Yuan, Xin and Ji, Jingyu and Zhang, Yifan and Wan, Shuai and Du, Qian},
TITLE = {Hyperspectral Image Spatial Super-Resolution via 3D Full Convolutional Neural Network},
JOURNAL = {Remote Sensing},
VOLUME = {9},
YEAR = {2017},
NUMBER = {11},
ARTICLE NUMBER = {1139},
URL = {http://www.mdpi.com/2072-4292/9/11/1139},
ISSN = {2072-4292},
DOI = {10.3390/rs9111139}
}
```
---
### Installation
##### Install Keras
1. Please follow [Tensorflow instruction](https://github.com/tensorflow/tensorflow) to install all necessary packages and build it.
2. Please follow [Keras instruction](https://github.com/keras-team/keras)

- Clone this repository. 
 - Note: We currently only support Python 2.7 

---
### Traineval

#####  datasets
1. download the datasets from [here](https://pan.baidu.com/s/1skW6MPv) and save in 'data' folder in data_process' folder
2. normalized and get mirrore of original data
```Shell
cd ./data_process
python expand.py
```
3. turn the data into small pieces
open your matlab and run gen_train_all_bands.m in 'data_process' folder

4. prepare the train data for keras
```Shell
cd ./data_process
python get_to_train.py
```
##### training
```Shell
python train_network.py
```
---
### Test

We release one pretrained models: model_pa.h5 in "model" folder for Pavia dataset. 
Do testing 
```Shell
python predict.py
```
will show both reconstructed images and PSNR/SSIM/SAM.

