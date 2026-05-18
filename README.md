# VisDrone 目标检测与多目标跟踪（YOLOv8 + ByteTrack）

本项目基于 YOLOv8 在 VisDrone 无人机航拍数据集上进行微调，得到专属检测模型，并利用 ByteTrack 算法对测试视频进行多目标跟踪，实现实时越线计数功能。

## 目录结构
.
├── YOLO.py # YOLOv8 微调训练脚本
├── test.py # 视频跟踪与越线计数脚本
├── best.pt # 微调后的最佳模型权重
├── test.mp4 # 测试视频（自行准备）
└── README.md # 本文件

text

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
conda create -n yolo_track python=3.10
conda activate yolo_track
2. 安装依赖
bash
pip install ultralytics wandb opencv-python
如果需要使用 GPU 加速，请确保已安装对应版本的 CUDA 和 PyTorch（ultralytics 会自动适配）。

数据集准备
本项目使用 VisDrone 数据集。训练脚本会自动使用 Ultralytics 内置的 VisDrone.yaml 配置文件，无需手动下载。首次运行时，脚本会自动下载并准备数据集。

若希望手动指定数据集路径，可以下载 VisDrone 数据集并修改 VisDrone.yaml 中的 path 字段。

训练模型
运行以下命令开始微调 YOLOv8n（基于 COCO 预训练权重）：

bash
python YOLO.py
训练参数在 YOLO.py 中配置：

参数	值	说明
epochs	50	训练轮数
batch	16	批量大小（可根据 GPU 显存调整）
imgsz	640	输入图像尺寸
device	0	使用 GPU 0（CPU 填 'cpu'）
project	VisDrone-Object-Detection	结果保存路径
name	yolov8n_visdrone_run	实验名称
训练过程中会自动记录指标到 wandb（需先 wandb login）。

训练完成后，最佳模型权重将保存在：

text
runs/detect/VisDrone-Object-Detection/yolov8n_visdrone_run3/weights/best.pt
（实际路径中的数字可能递增，请根据输出确认）

视频多目标跟踪与越线计数
使用训练好的模型对测试视频进行跟踪，并统计跨越虚拟线的目标数量。

修改测试脚本参数
编辑 test.py 中的以下变量：

python
MY_MODEL = "runs/detect/VisDrone-Object-Detection/yolov8n_visdrone_run3/weights/best.pt"   # 你的最佳模型路径
TEST_VIDEO = "test.mp4"          # 测试视频路径
可选：调整虚拟线位置（默认为画面高度的 65%）：

python
line_y = int(height * 0.65)
运行跟踪
bash
python test.py
程序将：

逐帧读取视频，利用 YOLOv8 + ByteTrack 进行检测与跟踪

为每个目标绘制边界框、Tracking ID 和类别标签

绘制虚拟计数线（红色）

实时更新跨越计数总数

输出带跟踪结果的视频 output_tracked.mp4

输出说明
控制台：打印发生跨越时的帧号、目标 ID 和类别

视频文件：output_tracked.mp4，包含所有可视化结果（边界框、ID、计数线、计数总数）

计数结果：视频左上角显示 Crossed Total: XX
