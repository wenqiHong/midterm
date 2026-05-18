import wandb
from ultralytics import YOLO

def main():
    # 1. 初始化 WandB 项目
    wandb.init(
        project="VisDrone-Object-Detection", 
        name="yolov8n_visdrone_run",
        config={
            "epochs": 50,
            "batch_size": 16,
            "imgsz": 640,
            "model": "yolov8n.pt"  # 使用预训练权重
        }
    )

    # 2. 加载在 ImageNet/COCO 上预训练的 YOLOv8 模型
    model = YOLO("yolov8n.pt") 

    # 3. 开始微调训练
    results = model.train(
        data="VisDrone.yaml",      # 数据集配置文件
        epochs=50,                 # 训练轮数
        imgsz=640,                 # 输入图像尺寸
        batch=16,                  # Batch size
        workers = 1,
        device=0,                 
        project="VisDrone-Object-Detection",
        name="yolov8n_visdrone_run",
        val=True                   # 训练过程中自动在验证集上进行评估
    )
    
    # 4. 结束 WandB 进程
    wandb.finish()

if __name__ == "__main__":
    main()