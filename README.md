# CGRNet: Contour-Guided Graph Reasoning Network for Ambiguous Biomedical Image Segmentation
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
## 0. Preface
- [2022/01/5]:**Submitted to the journal of "BSPC " （Under Review）**


### 1.1. 🔥NEWS🔥 :
- [2021/10/30]:fire: Release the inference code!
- [2021/10/28] Create repository.


## Prerequisites
- [Python 3.5](https://www.python.org/)
- [Pytorch 1.1](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy 1.15](https://numpy.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)

## Clone repository
```shell
git clone https://github.com/DLWK/CGRNet.git
cd CGRNet/
```
## Download dataset
Download the datasets and unzip them into `data` folder
- [COVID-19](https://medicalsegmentation.com/covid19/)
## Training & Evaluation
```shell
 cd CGRNet/
 python3 train.py
 ################
 python3 test.py
 
```

## Demon
```shell
from  CGRmoes.CGR import  CGRNet
if __name__ == '__main__':
    ras =CGRNet(n_channels=3, n_classes=1).cuda()
    input_tensor = torch.randn(4, 3, 352, 352).cuda()

    out,out1 = ras(input_tensor)
    print(out.shape)
   
 
```
# Tips
:fire:If you have any questions about our work, please do not hesitate to contact us by emails.
**[⬆ back to top](#0-preface)**
