from ultralytics import YOLO
 
# 加载预训练的YOLOv11n模型
model = YOLO(r"yolo11x-seg.pt")
 
# 对指定的图像文件夹进行推理，并设置各种参数
results = model.predict(
    source="data/coco/train2017/",   # 新的图片，待标注的图片
    project="runs/segment",  # 主输出目录名称
    name="trainpredict",          # 子文件夹名称（自动创建）
    conf=0.2,                      # 置信度阈值
    iou=0.4,                        # IoU 阈值
    imgsz=1024,                      # 图像大小
    half=True,                     # 使用半精度推理
    device=None,                    # 使用设备，None 表示自动选择，比如'cpu','0'
    max_det=300,                    # 最大检测数量
    vid_stride=1,                   # 视频帧跳跃设置
    stream_buffer=False,            # 视频流缓冲
    visualize=False,                # 可视化模型特征
    augment=False,                  # 启用推理时增强
    agnostic_nms=False,             # 启用类无关的NMS
    classes=[
    4, 1, 14, 8, 39,
    5, 2, 15, 56, 19,
    60, 16, 17, 3, 0,
    58, 18, 57, 6, 62
],  # 0~19对应VOC类别
    retina_masks=True,             # 使用高分辨率分割掩码
    embed=None,                     # 提取特征向量层
    show=False,                     # 是否显示推理图像
    save=True,                      # 保存推理结果
    save_frames=False,              # 保存视频的帧作为图像
    save_txt=True,                  # 保存检测结果到文本文件
    save_conf=True,                # 保存置信度到文本文件
    save_crop=False,                # 保存裁剪的检测对象图像
    show_labels=True,               # 显示检测的标签
    show_conf=True,                 # 显示检测置信度
    show_boxes=True,                # 显示检测框
    line_width=2                    # 设置边界框的线条宽度，比如2，4
)