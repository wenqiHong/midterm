import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def run_tracking_and_counting(video_path, model_path, output_path="output_tracked.mp4"):
    # 1. 加载微调后的专属检测模型
    model = YOLO(model_path)

    # 2. 读取测试视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return

    # 获取视频的基本属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 3. 设置视频输出写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 设定虚拟越线（例如：横向穿过视频画面 65% 高度处的一条红线）
    line_y = int(height * 0.65)

    # 5. 初始化跟踪历史和计数变量
    track_history = defaultdict(lambda: [])
    counted_ids = set()  # 记录已经被计过数的 ID，防止目标在红线附近震荡时重复计数
    cross_count = 0

    print("开始处理视频流...")
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1

        # 使用 YOLOv8 内置的 ByteTrack 算法进行多目标跟踪
        # persist=True 确保模型在帧与帧之间保持跟踪状态
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        # 检查当前帧是否有检测并跟踪到的目标
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = box
                # 计算检测框的中心点坐标
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 获取该 ID 的历史运动轨迹
                prev_positions = track_history[track_id]
                
                if len(prev_positions) > 0:
                    prev_cx, prev_cy = prev_positions[-1]
                    
                    # 越线逻辑判断：前一帧在红线上方，当前帧在红线下方（或反之）
                    if (prev_cy < line_y <= cy) or (cy <= line_y < prev_cy):
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            cross_count += 1
                            print(f"[Frame {frame_idx}] 目标 ID:{track_id} ({model.names[cls]}) 跨越了计数线！")

                # 更新当前 ID 的轨迹历史
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:  # 限制轨迹长度防止内存占用过大
                    track_history[track_id].pop(0)

                # 绘制目标边界框、Tracking ID 和类别标签
                label = f"ID:{track_id} {model.names[cls]}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 绘制中心点
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # 6. 绘制虚拟计数线和实时计数结果
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(frame, f"Crossed Total: {cross_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 7. 将处理后的帧写入输出视频
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"视频处理完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    MY_MODEL = "runs/detect/VisDrone-Object-Detection/yolov8n_visdrone_run3/weights/best.pt" 
    TEST_VIDEO = "test.mp4" 
    
    run_tracking_and_counting(TEST_VIDEO, MY_MODEL)