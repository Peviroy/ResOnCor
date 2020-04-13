<!-- PROJECT -->
<br />

  <h3 align="center">ResOnCor</h3>

  <p align="center">
    Resnet implement on Corel-10 dataset
    <br />
    <a href="https://github.com/Peviroy/ResOnCor"><strong>Explore the docs »</strong></a>
    <br />



</p>

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



## About The Project

此项目诞生于模式识别的一次作业。总的来说是以resnet18为网络骨干实现了在Corel-1000数据集上面的图像分类功能，并且能够达到98%左右的训练准确率，以及较为跌宕但总体在95%左右的测试准确率。（这或许同测试集本身有所关联）

诚然resnet的架设是本次的一大重点，也着实花费了相当精力去借鉴、去实现。但我们并不希望止步于此，毕竟单有网络架构以及基本的训练代码是不足以称之为一个项目的。出于易用性以及可维护性的考虑，在整体代码风格以及层次搭建上，我们花费了最多的精力，约占总体耗时的三分之二以上。

### Directory tree

```bash
.
├── checkpoints					# saving the resulting models
├── data						# [python directory]Code to generate the data set
│   ├── __init__.py
│   └── dataset.py
├── dataset					
│   ├── test
│   │   ├── African
│   │   ├── ........
│   │   └── test.txt			# test namefile
│   └── train
│       ├── African
│       ├── .......
│       └── test.txt			# train namefile
├── models						# [python directory]model defination
│   ├── __init__.py
│   └── Resnet.py
└── utils						# [python directory]utility
│   ├── __init__.py
│   ├── util.py					
│   ├── torch_util.py
│   └── Meter.py				# Special dashboard for recording network output.
├── __init__.py
├── main.py						# entry file
└── requirement.txt
```


### Built With

* [Pytorch](https://github.com/pytorch/pytorch)

### Requirement 

* matplotlib
* numpy
* torchvision
* opencv_python # This package has only been used in insignificant places and can be ignored.
* torch
* Pillow

#### versions  used

​	There is no feature code in this project, so any release version is ok. Here is my version below:

```
matplotlib==3.2.1
numpy==1.16.4
torchvision==0.5.0
opencv_python==4.2.0.34
torch==1.4.0
Pillow==7.1.1
```



## Getting Started

### Prerequisites

Python virtual environment necessary to reproduce. Here [anaconda](https://www.anaconda.com/) is recommended. However, the virtual environment is not indispensable, any environment is welcome, as long as the environment can be built
* conda
```sh
conda create -n new_clean_env python=3.7
conda activate new_clean_env
```

### Installation

1. Clone the repo
```sh
git https://github.com/Peviroy/ResOnCor.git
```
2. Install requirements
```sh
cd ResOnCor
pip install -r requirements 
```

​		if internet speed is not ok, try switching mirror source or using [proxychains](https://github.com/haad/proxychains) for commandline proxy

## Training

To train a model, run `main.py` :

```
python main.py 
```

The default learning rate schedule starts at 0.01. This may be somewhat lower than usaual, but can converge steadily.

For more augment infomation, call `--help`:

```bash
python main.py --help            

usage: main.py [-h] [--model-folder MODEL_FOLDER] [--data DATA]
               [--batch BATCH] [--epoch EPOCH] [--save SAVE] [--lr LR]
               [--momentum MOMENTUM] [--gpu GPU] [--weight-decay WEIGHT_DECAY]

Resnet on CorelDataset

optional arguments:
  -h, --help            show this help message and exit
  --model-folder MODEL_FOLDER
                        folder to save models
  --data DATA           where the data set is stored
  --batch BATCH         batch size of data input(default: 64)
  --epoch EPOCH         the number of cycles to train the model(default: 200)
  --save SAVE           dir for saving document file
  --lr LR               learning rate(default: 0.01)
  --momentum MOMENTUM   momentum(default: 0.9)
  --gpu GPU             GPU id to use
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 5e-4)
```





### TODO:

- [ ] Add lr decay function;

- [ ] Pre-train mode

- [ ] Validation mode
- [ ] CNN visualization

## Contact

- [email](https://twitter.com/twitter_handle) - peviroy@outlook.com

Project Link: [https://github.com/Peviroy/ResOnCor](https://github.com/Peviroy/ResOnCor)



## Acknowledgements

* [Official implement of resnet ](https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet)
* [ImageNet training in PyTorch](https://github.com/pytorch/examples/blob/master/imagenet/main.py)


