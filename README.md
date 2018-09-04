# 眼底水肿病变区域自动分割

视网膜水肿是一种眼部疾病，严重时会导致视力下降从而影响正常的生活。现在医学使用OCT（光学相干断层成像）辅助医生对视网膜水肿的判断。尽早的发现水肿症状，能够对疾病的治疗起到关键性作用。


本仓库基于用SegNet进行语义分割。

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/fundus_lesion_eg1.png)

按照 [说明](https://challenger.ai/competition/fl2018) 下载 眼底水肿病变区域自动分割 数据集，放在 data 目录内。

## 架构

![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/segnet.png)

## 用法
### 数据预处理
该数据集包含SUNRGBD V1的10335个RGBD图像，执行下述命令提取训练图像：
```bash
$ python pre-process.py
```

像素分布：

![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/dist.png)

### 数据集增强
图片 | 分割 | 图片 | 分割 |
|---|---|---|---|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_category_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_category_aug.png) |
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_category_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_category_aug.png) |
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_category_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_category_aug.png) |
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_category_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_category_aug.png) |
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_category_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_image_aug.png) |![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_category_aug.png) |

### 训练
```bash
$ python train.py
```

如果想可视化训练过程，可执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/learning_curve.png)

### 演示

下载 [预训练模型](https://github.com/foamliu/Semantic-Segmentation/releases/download/v1.0/model.81-3.5244.hdf5) 放在 models 目录，然后执行:


```bash
$ python demo.py
```

图例：

![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/legend.png)

输入 | 真实 | 输出 |
|---|---|---|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_out.png)|

