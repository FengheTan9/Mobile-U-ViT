# Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient medical image segmentation

![Teaser](imgs/teaser.jpg)



<div align="center">
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=x1pODsMAAAAJ&hl=en" target="_blank">Fenghe Tang</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a target="_blank">Bingkun Nian</a><sup>3</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=4TsvOR8AAAAJ&hl=en" target="_blank">Jianrui Ding</a><sup>4</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=r0-tZ8cAAAAJ&hl=en" target="_blank">Wenxin Ma</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=mlTXS0YAAAAJ&hl=en" target="_blank">Quan Quan</a><sup>5</sup>,</span>
    <br>
    <span class="author-block">
    <a target="_blank">Chengqi Dong</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=tmx7tu8AAAAJ&hl=en" target="_blank">Jie Yang</a><sup>3</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=Vbb5EGIAAAAJ&hl=en" target="_blank">Wei Liu</a><sup>3</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ&hl=en" target="_blank">S. Kevin Zhou</a><sup>1,2</sup>
    </span>
</div>
<br>


<div align="center">
    <sup>1</sup>
    <a href='https://en.ustc.edu.cn/' target='_blank'>School of Biomedical Engineering, University of Science and Technology of China</a>&emsp;
    <br>
    <sup>2</sup> <a href='http://english.ict.cas.cn/' target='_blank'>Suzhou Institute for Advanced Research, University of Science and Technology of China</a>&emsp;
    <br>
    <sup>3</sup> <a href='http://www.pami.sjtu.edu.cn/En/Home' target='_blank'>School of Automation and Intelligent Sensing, Shanghai Jiao Tong University</a>
    <br>
    <sup>4</sup> <a href='https://en.hit.edu.cn/' target='_blank'>School of Computer Science and Technology, Harbin Institute of Technology</a>
    <br>
    <sup>5</sup> <a>State Grid Hunan ElectricPower Corporation Limited Research Institute</a>
</div>
<br>

   [![arXiv](https://img.shields.io/badge/arxiv-2508.01064-b31b1b)](https://arxiv.org/pdf/2508.01064.pdf)   [![github](https://img.shields.io/badge/github-MobileUViT-black)](https://github.com/FengheTan9/Mobile-U-ViT)    <a href="####License"><img alt="License: Apache2.0" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue.svg"/></a>



## News

- **Mobile U-ViT accepted by ACM MM'25 🥰** 
- **Paper and Code released !** 😎

### Abstract

In clinical practice, medical image analysis often requires efficient execution on resource-constrained mobile devices. However, exist ing mobile models—primarily optimized for natural images—tend to perform poorly on medical tasks due to the significant information density gap between natural and medical domains. Combining com putational efficiency with medical imaging-specific architectural advantages remains a challenge when developing lightweight, uni versal, and high-performing networks. To address this, we propose a mobile model called Mobile U-shaped Vision Transformer (Mobile U-ViT) tailored for medical image segmentation. Specifically, we employ the newly purposed ConvUtr as a hierarchical patch embedding, featuring a parameter-efficient large-kernel CNN with inverted bottleneck fusion. This design exhibits transformer-like representation learning capacity while being lighter and faster. To enable efficient local-global information exchange, we introduce a novel Large-kernel Local-Global-Local (LGL) block that effectively balances the low information density and high-level semantic dis crepancy of medical images. Finally, we incorporate a shallow and lightweight transformer bottleneck for long-range modeling and employ a cascaded decoder with downsample skip connections for dense prediction. Despite its reduced computational demands, our medical-optimized architecture achieves state-of-the-art per formance across eight public 2D and 3D datasets covering diverse imaging modalities, including zero-shot testing on four unseen datasets. These results establish it as an efficient yet powerful and generalization solution for mobile medical image analysis.

![Teaser](imgs/network.jpg)

### Results:

![Teaser](imgs/compare.jpg)

![Teaser](imgs/analysis.jpg)

# Quick Start

#### 1. Environment

- GPU: NVIDIA GeForce RTX4090 GPU
- Pytorch: 1.13.0 cuda 11.7
- cudatoolkit: 11.7.1
- scikit-learn: 1.0.2
- albumentations: 1.2.0

#### 2. Datasets

Please put the [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) dataset or your own dataset as the following architecture. 
```
└── Mobile-U-ViT
    ├── data
        ├── busi
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── ...
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── ...
    ├── dataloader
    ├── network
    ├── utils
    ├── main.py
    └── split.py
```
#### 3. 2D Training & Validation

You can first split your dataset:

```python
python split.py --dataset_name busi --dataset_root ./data
```

Then, train and validate:

```python
python main.py --model ["mobileuvit", "mobileuvit_l"] --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
```
#### 4. 3D Training & Validation

Downstream pipeline can be referred to [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV).

```python
# An example of Training on BTCV (num_classes=14)
from network.MobileUViT_3D import mobileuvit_l

model = mobileuvit_l(inch=1, out_channel=14).cuda()
```

![Teaser](imgs/result.jpg)

### Acknowledgements:

This code uses helper functions from [CMUNeXt](https://github.com/FengheTan9/CMUNeXt).

#### Citation

If the code, paper and weights help your research, please cite:

```
@article{tang2025mobile,
  title={Mobile U-ViT: Revisiting large kernel and U-shaped ViT for efficient medical image segmentation},
  author={Tang, Fenghe and Nian, Bingkun and Ding, Jianrui and Ma, Wenxin and Quan, Quan and Dong, Chengqi and Yang, Jie and Liu, Wei and Zhou, S Kevin},
  journal={arXiv preprint arXiv:2508.01064},
  year={2025}
}
```

#### License

This project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
