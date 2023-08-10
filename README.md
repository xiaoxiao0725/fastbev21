+ [环境部署参考](https://blog.csdn.net/h904798869/article/details/130317240?spm=1001.2014.3001.5502)

---

# Fast-BEV

~~~
python setup.py clean
mim install --no-cache-dir -e -v .
~~~


~~~python
mmcv-full                     1.4.0
mmdet                         2.14.0
mmdet3d                       0.16.0 
mmsegmentation                0.14.1
~~~

~~~
tools/test.py configs/fastbev/fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.py epoch_20.pth --eval bbox
~~~


~~~
pip install future tensorboard
~~~

~~~
AttributeError: module 'distutils' has no attribute 'version' 
conda install setuptools==58.0.4
~~~

## 制作mini数据集

1. 生成数据集mini
~~~
python tools/create_data.py --dataset nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --workers 10 --version v1.0-mini
~~~

+ 这个参数一定要给--extra-tag nuscenes 初始化info_prefix=args.extra_tag

2. 产生nuscenes_infos_train_4d_interval3_max60.pkl与nuscenes_infos_test_4d_interval3_max60.pkl用于训练

~~~
python tools/data_converter/nuscenes_seq_converter.py
~~~

+ mini数据集修改nuscenes_seq_converter.py的15行代码，因为mini无测试集


3. test
~~~python
python tools/test.py
~~~

3. pkl检测结果生成视频可视化

~~~python
python tools/misc/visualize_results.py
~~~

---

[Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline](https://arxiv.org/abs/2301.12511)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/fast-bev++.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark_setting.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark.png)

## Usage

[usage](https://github.com/Sense-GVT/Fast-BEV/blob/dev/tools/fastbev_run.sh)

### Installation

* CUDA>=9.2
* GCC>=5.4
* Python>=3.6
* Pytorch>=1.8.1
* Torchvision>=0.9.1
* MMCV-full==1.4.0
* MMDetection==2.14.0
* MMSegmentation==0.14.1


### Dataset preparation

```
  .
  ├── data
  │   └── nuscenes
  │       ├── maps
  │       ├── maps_bev_seg_gt_2class
  │       ├── nuscenes_infos_test_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_train_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_val_4d_interval3_max60.pkl
  │       ├── v1.0-test
  │       └── v1.0-trainval
```

[download](https://drive.google.com/drive/folders/10KyLm0xW3QiLhAefxBbXR-Hw_7nel_tm?usp=sharing)

### Pretraining

```
  .
  ├── pretrained_models
  │   ├── cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth
  │   ├── cascade_mask_rcnn_r34_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5190_segm_mAP_0.4140.pth
  │   └── cascade_mask_rcnn_r50_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5400_segm_mAP_0.4300.pth
```

[download](https://drive.google.com/drive/folders/19BD4totDHtwnHtOqTdn0xYJh7stwYd9l?usp=sharing)

### Training

```
  .
  ├── work_dirs
    └── fastbev
      └── exp
          └── paper
              └── fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.02062323.txt
              │   └── log.test.fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.02062309.txt
              ├── fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4.02080000.txt
              │   └── log.test.fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4.02072346.txt
              ├── fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.02080021.txt
              │   └── log.test.fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.02080005.txt
              ├── fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.02080021.txt
              │   └── log.test.fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.02080005.txt
              └── fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4
                  ├── epoch_20.pth
                  ├── latest.pth -> epoch_20.pth
                  ├── log.eval.fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.02080021.txt
                  └── log.test.fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.02080001.txt
```

[download](https://drive.google.com/drive/folders/1Ja9mqOE0iGPysVxmLSrZyUoCEBYu5fMH?usp=sharing)

### Deployment
TODO

## View Transformation Latency on device
[2D-to-3D on CUDA & CPU](https://github.com/Sense-GVT/Fast-BEV/tree/dev/script/view_tranform_cuda)

## Citation
```
@article{li2023fast,
  title={Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline},
  author={Li, Yangguang and Huang, Bin and Chen, Zeren and Cui, Yufeng and Liang, Feng and Shen, Mingzhu and Liu, Fenggang and Xie, Enze and Sheng, Lu and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2301.12511},
  year={2023}
}
```
