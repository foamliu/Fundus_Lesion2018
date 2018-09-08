# 眼底水肿病变区域自动分割

视网膜水肿是一种眼部疾病，严重时会导致视力下降从而影响正常的生活。现在医学使用OCT（光学相干断层成像）辅助医生对视网膜水肿的判断。尽早的发现水肿症状，能够对疾病的治疗起到关键性作用。


本仓库基于用SegNet进行语义分割。

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

自动分割OCT视网膜图像中的三种水肿病变区域：视网膜色素上皮脱离（PED）、视网膜下水肿（SRF）和视网膜水肿区域（REA）:

![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/fundus_lesion_eg1.png)

按照 [说明](https://challenger.ai/competition/fl2018) 下载 眼底水肿病变区域自动分割 数据集，放在 data 目录内。

## 架构

![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/segnet.png)

## 用法
### 数据预处理
执行下述命令提取训练图像(训练集：70x128=8960 张，验证集：15x128=1920 张，测试集：15x128=1920 张)：
```bash
$ python pre_process.py
```

### 训练
```ba$ python train.py
```

如果想可视化训练过程，可执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 演示

下载 [预训练模型](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/releases/download/v1.0/model.120-0.7736.hdf5) 放在 models 目录，然后执行:


```bash
$ python demo.py
```

输入 | 输出 | 输入 | 输出 |
|---|---|---|---|
|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/0_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/0_out.png)|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/1_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/1_out.png)|
|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/2_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/2_out.png)|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/3_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/3_out.png)|
|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/4_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/4_out.png)|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/5_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/5_out.png)|
|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/6_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/6_out.png)|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/7_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/7_out.png)|
|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/8_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/8_out.png)|![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/9_image.png)| ![image](https://github.com/foamliu/Automatic-Segmentation-Of-Fundus-Edema-Lesions/raw/master/images/9_out.png)|

