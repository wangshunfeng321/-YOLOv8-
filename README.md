# 基于 YOLOv8 的目标检测系统

## 项目简介

本项目是一个基于 Python、OpenCV 和 YOLOv8 的目标检测系统，支持图片目标检测、摄像头实时目标检测以及 ONNX 模型部署测试。项目通过调用 YOLOv8 预训练模型，实现对图像或视频流中目标的识别、检测框绘制、类别显示、置信度显示和 FPS 实时统计。

本项目适合作为深度学习目标检测入门实践项目，用于理解 YOLO 模型推理流程、OpenCV 图像处理方法以及模型轻量化部署的基本过程。

## 项目功能

* 支持单张图片目标检测
* 支持摄像头实时视频目标检测
* 支持 YOLOv8 预训练模型加载与推理
* 支持检测框、类别名称、置信度可视化显示
* 支持实时 FPS 显示
* 支持将 YOLOv8 模型导出为 ONNX 格式
* 支持基于 OpenCV DNN 的 ONNX 模型推理测试
* 提供交互式菜单，方便选择不同检测模式

## 技术栈

* Python 3.11
* OpenCV
* Ultralytics YOLOv8
* PyTorch
* NumPy
* ONNX
* OpenCV DNN

## 项目结构

```text
YOLOv8-Object-Detection/
│
├── 实验二 目标检测.py        # 主程序文件
├── yolov8n.pt               # YOLOv8 预训练模型
├── yolov8n.onnx             # 导出的 ONNX 模型
├── test.png                 # 测试图片
├── assets/                  # 项目运行截图
│   ├── image_detection.png
│   ├── realtime_detection.png
│   └── onnx_detection.png
└── README.md                # 项目说明文档
```

## 环境安装

首先安装项目所需依赖：

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install torch torchvision
```

也可以将依赖写入 `requirements.txt`：

```text
ultralytics
opencv-python
numpy
torch
torchvision
```

然后执行：

```bash
pip install -r requirements.txt
```

## 运行方式

运行主程序：

```bash
python "实验二 目标检测.py"
```

程序启动后，会出现功能菜单：

```text
请选择功能:
1. 图像目标检测（YOLOv8）
2. 实时视频检测（YOLOv8）
3. 优化模型实时检测（OpenCV DNN + ONNX）
4. 退出
```

### 1. 图片目标检测

选择功能 `1`，输入图片路径，例如：

```text
C:\Users\24576\Desktop\lianxi\实验二\test.png
```

程序会对图片中的目标进行检测，并显示目标类别、置信度和检测框。

### 2. 摄像头实时检测

选择功能 `2`，输入：

```text
0
```

即可调用电脑摄像头进行实时检测。程序会在检测窗口中显示目标框、类别、置信度和 FPS。

按 `q` 键可以退出检测窗口。

### 3. ONNX 模型部署检测

选择功能 `3`，程序会加载 `yolov8n.onnx` 模型，并使用 OpenCV DNN 进行 CPU 端推理检测。

该功能主要用于测试 YOLOv8 模型导出为 ONNX 后的部署流程，以及不同推理方式下的运行效果。

## 项目运行效果

### 图片目标检测

模型成功识别测试图片中的小狗目标，并输出检测类别和置信度。

![图片目标检测](assets/image_detection.png)

### 摄像头实时检测

模型能够实时检测摄像头画面中的人物和椅子等目标，并显示 FPS。

![实时目标检测](assets/realtime_detection.png)

### ONNX 模型检测

将 YOLOv8 模型导出为 ONNX 格式后，使用 OpenCV DNN 完成 CPU 端推理测试。

![ONNX模型检测](assets/onnx_detection.png)

## 实验结果

本项目完成了 YOLOv8 目标检测系统的基本搭建，并实现了以下测试：

| 测试内容      | 测试结果                             |
| --------- | -------------------------------- |
| 图片目标检测    | 能够识别图片中的 dog 目标，并显示检测框和置信度       |
| 摄像头实时检测   | 能够检测 person、chair 等目标，并显示实时 FPS  |
| ONNX 模型部署 | 成功加载 ONNX 模型，并使用 OpenCV DNN 完成推理 |
| 交互式运行     | 能够通过菜单选择不同检测模式                   |

在摄像头实时检测测试中，YOLOv8 模型能够在 640×480 视频流下进行实时检测，运行帧率约为 23 FPS。ONNX + OpenCV DNN 模式完成了 CPU 端推理验证，但实际帧率较低，后续可继续进行性能优化。

## 核心代码说明

项目主要包含三个检测类：

### ObjectDetector

用于图片目标检测，主要功能包括：

* 加载 YOLOv8 模型
* 输入图片路径
* 执行目标检测
* 绘制检测框、类别名称和置信度

### RealtimeDetector

用于摄像头或视频流实时目标检测，主要功能包括：

* 调用摄像头或视频文件
* 对每一帧图像进行检测
* 实时显示检测结果
* 计算并显示 FPS

### OptimizedDetector

用于 ONNX 模型推理测试，主要功能包括：

* 将 YOLOv8 模型导出为 ONNX 格式
* 使用 OpenCV DNN 加载 ONNX 模型
* 完成图像预处理、模型推理和结果后处理
* 进行 CPU 端部署测试

## 项目收获

通过本项目，我熟悉了 YOLOv8 目标检测模型的基本使用流程，掌握了图像检测、视频流检测、检测结果可视化和 FPS 性能统计方法。同时，对模型从 PyTorch 格式导出为 ONNX 格式，以及使用 OpenCV DNN 进行轻量化部署有了初步理解。

## 后续优化方向

* 优化 ONNX 推理速度，提高实时检测帧率
* 增加检测结果保存功能
* 支持视频文件检测结果导出
* 增加图形化界面，提高项目易用性
* 使用自定义数据集训练模型，提高特定场景下的检测效果
* 尝试使用 GPU、TensorRT 等方式进一步提升推理速度

## 项目总结

本项目实现了一个完整的 YOLOv8 目标检测实验流程，包括模型加载、图片检测、摄像头实时检测、ONNX 模型导出和 OpenCV DNN 推理测试。项目结构清晰，功能完整，适合作为深度学习目标检测方向的基础实践项目。
