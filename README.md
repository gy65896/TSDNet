 # <p align=center> [TII 2022] Deep Network-Enabled Haze Visibility Enhancement for Visual IoT-Driven Intelligent Transportation Systems</p>

<div align="center">
 
[![Paper](https://img.shields.io/badge/TSDNet-Paper-red.svg)](https://ieeexplore.ieee.org/abstract/document/9764372)

</div>

---
>**Deep Network-Enabled Haze Visibility Enhancement for Visual IoT-Driven Intelligent Transportation Systems**<br>  [Ryan Wen Liu](http://mipc.whut.edu.cn/index.html)<sup>* </sup>, [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN), [Yuxu Lu](https://scholar.google.com.hk/citations?user=XXge2_0AAAAJ&hl=zh-CN), Kwok Tai Chui, Brij B Gupta <br>
(* Corresponding Author)<br>
>IEEE Transactions on Industrial Informatics

> **Abstract:** *The Internet of Things (IoT) has recently emerged as a revolutionary communication paradigm where a large number of objects and devices are closely interconnected to enable smart industrial environments. The tremendous growth of visual sensors can significantly promote the traffic situational awareness, traffic safety management, and intelligent vehicle navigation in intelligent transportation systems (ITS). However, due to the absorption and scattering of light by the turbid medium in atmosphere, the visual IoT inevitably suffers from imaging quality degradation, e.g., contrast reduction, color distortion, etc. This negative impact can not only reduce the imaging quality, but also bring challenges for the deployment of several high-level vision tasks (e.g., object detection, tracking and recognition, etc.) in ITS. To improve imaging quality under the hazy environment, we propose a deep network-enabled three-stage dehazing network (termed TSDNet) for promoting the visual IoT-driven ITS. In particular, the proposed TSDNet mainly contains three parts, i.e., multi-scale attention module for estimating the hazy distribution in the RGB image domain, two-branch extraction module for learning the hazy features, and multi-feature fusion module for integrating all characteristic information and reconstructing the haze-free image. Numerous experiments have been implemented on synthetic and real-world imaging scenarios. Dehazing results illustrated that our TSDNet remarkably outperformed several state-of-the-art methods in terms of both qualitative and quantitative evaluations. The high-accuracy object detection results have also demonstrated the superior dehazing performance of TSDNet under hazy atmosphere conditions.*
---

## Requirement
* __Python__ == 3.7
* __Pytorch__ == 1.9.1

## Network Architecture
![Figure02_Flowchart](https://user-images.githubusercontent.com/48637474/233028445-4e0a0e3e-7e32-4833-943f-80f144a6d28f.jpg)

## Test
* Put the hazy image in the `input` folder
* Run `test_real.py`. 
* The enhancement result will be saved in the `result` folder.

## Citation

```
@article{liu2022deep,
  title={Deep network-enabled haze visibility enhancement for visual IoT-driven intelligent transportation systems},
  author={Liu, Ryan Wen and Guo, Yu and Lu, Yuxu and Chui, Kwok Tai and Gupta, Brij B},
  journal={IEEE Transactions on Industrial Informatics},
  volume={19},
  number={2},
  pages={1581--1591},
  year={2022},
  publisher={IEEE}
}
```

#### If you have any questions, please get in touch with me (guoyu65896@gmail.com).
