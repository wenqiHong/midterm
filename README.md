# VisDrone 目标检测与多目标跟踪（YOLOv8 + ByteTrack）

本项目基于 YOLOv8 在 VisDrone 无人机航拍数据集上进行微调，得到专属检测模型，并利用 ByteTrack 算法对测试视频进行多目标跟踪，实现实时越线计数功能。

## 目录结构
YOLO.py # YOLOv8 微调训练脚本

test.py # 视频跟踪与越线计数脚本

best.pt # 微调后的最佳模型权重

test.mp4 # 测试视频（自行准备）

README.md # 本文件


## 数据集准备
本项目使用 VisDrone 数据集。训练脚本会自动使用 Ultralytics 内置的 VisDrone.yaml 配置文件，无需手动下载。首次运行时，脚本会自动下载并准备数据集。

若希望手动指定数据集路径，可以下载 VisDrone 数据集并修改 VisDrone.yaml 中的 path 字段。

## 训练模型
运行以下命令开始微调 YOLOv8n：

python YOLO.py

训练完成后，最佳模型权重将保存在：
runs/detect/VisDrone-Object-Detection/yolov8n_visdrone_run/weights/best.pt 

#可能会随着训练次数生成yolov8n_xisdrone_run2,3,4等，需要相应修改test.py中的路径



## 输出说明
控制台：打印发生跨越时的帧号、目标 ID 和类别

视频文件：output_tracked.mp4，包含所有可视化结果（边界框、ID、计数线、计数总数）

计数结果：视频左上角显示 Crossed Total: XX
