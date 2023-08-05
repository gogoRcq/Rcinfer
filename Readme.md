## RcInfer

#### 简介

这是一个简单的 C++推理框架，目前支持了 Restnet 模型的推理（用于图像分类），后续可能会进一步支持 YOLO模型的推理。学习自：https://github.com/zjhellofss/KuiperInfer 。

#### 依赖库

该推理框架的算子都是在 CPU 上做计算的，并且利用了 glog，google test 等库，以下是使用本框架所需要安装的依赖：

- **Armadillo**数学库，官方网站：https://arma.sourceforge.net/docs.html
- **OpenBlas**线性代数库
- ***Lapack***线性代数库
- **gtest**单元测试库
- **glog** 日志库
- **opencv**计算机视觉库

#### 文件目录介绍

`include`文件夹中包含了本项目的头文件，`src`文件夹中包含了框架的 cpp 文件，`src`文件夹中 `layer`文件中有算子的具体实现。`test`文件夹中包含了测试用例，有对算子单独测试的用例，也有对模型推理的用例。`tmp`文件夹是测试所用到的一些文件。

#### 运行

编译好之后，执行测试用例即可看到测试结果。

