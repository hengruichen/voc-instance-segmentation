import os
import json
import cv2
 
# 定义VOC类别标签（顺序与您的JSON完全一致）
LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# 定义COCO ID到VOC ID的映射（根据图片内容整理）
COCO_TO_VOC = {
    # 人物类
    0: 14,   # person → person
    
    # 交通工具类
    1: 1,    # bicycle → bicycle
    2: 6,    # car → car
    5: 5,    # bus → bus
    6: 18,   # train → train
    4: 0,    # airplane → aeroplane
    8: 3,    # boat → boat
    3: 13,   # motorcycle → motorbike
    
    # 动物类
    14: 2,   # bird → bird
    15: 7,   # cat → cat
    16: 11,  # dog → dog
    17: 12,  # horse → horse
    18: 16,  # sheep → sheep
    19: 9,   # cow → cow
    
    # 家具/物品类
    56: 8,   # chair → chair
    57: 17,  # couch → sofa
    58: 15,  # potted plant → pottedplant
    60: 10,  # dining table → diningtable
    62: 19,  # tv → tvmonitor
    # 其他可对应类别
    39: 4,   # bottle → bottle
    6: 18,   # truck → train (近似映射)
}

def convert_coco_to_voc(coco_id):
    """将COCO ID转换为VOC标签"""
    voc_id = COCO_TO_VOC.get(coco_id)
    return LABELS[voc_id] if voc_id is not None else None
 
def yolo11_to_labelme(txt_file, img_file, save_dir, labels):
    """
    将YOLO11格式的分割标签文件转换为Labelme格式的JSON文件。
    参数：
    - txt_file (str): YOLO11标签的txt文件路径。
    - img_file (str): 对应的图像文件路径。
    - save_dir (str): JSON文件保存目录。
    - labels (list): 类别标签列表。
    """
    # 读取图像，获取图像尺寸
    img = cv2.imread(img_file)
    height, width, _ = img.shape
 
    # 创建Labelme格式的JSON数据结构
    labelme_data = {
        "version": "4.5.9",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_file),
        "imageHeight": height,
        "imageWidth": width,
        "imageData": None  # 可以选择将图像数据转为base64后嵌入JSON
    }
 
    # 读取YOLO11标签文件
    with open(txt_file, "r") as file:
        for line in file.readlines():
            data = line.strip().split()
            class_id = int(data[0])  # 类别ID
            points = list(map(float, data[1:]))  # 获取多边形坐标
            #print(points)
            # 确保坐标点数量是偶数
            if len(points) % 2 != 0:
                print(f"警告：坐标点数量不是偶数，将去掉最后一个点: {txt_file}")
                points = points[:-1]  # 去掉最后一个点
            # 转换类别ID到VOC标签
            voc_label = convert_coco_to_voc(class_id)
            if voc_label is None:
                print(f"跳过无对应VOC类别的COCO ID: {class_id}")
                continue
            # 将归一化坐标转换为实际像素坐标
            polygon = []
            for i in range(0, len(points), 2):
                x = points[i] * width
                y = points[i + 1] * height
                polygon.append([x, y])
 
            # 定义多边形区域
            shape = {
                "label": voc_label,  # 使用直接定义的类别名称
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",  # 分割使用多边形
                "flags": {}
            }
            labelme_data["shapes"].append(shape)
 
    # 保存为labelme格式的JSON文件
    save_path = os.path.join(save_dir, os.path.basename(txt_file).replace(".txt", ".json"))
    with open(save_path, "w") as json_file:
        json.dump(labelme_data, json_file, indent=4)
 
def convert_yolo11_to_labelme(txt_folder, img_folder, save_folder):
    """
    读取文件夹中的所有txt文件，将YOLO11标签转为Labelme的JSON格式。
    参数：
    - txt_folder (str): 存放YOLO11 txt标签文件的文件夹路径。
    - img_folder (str): 存放图像文件的文件夹路径。
    - save_folder (str): 保存转换后的JSON文件的文件夹路径。
    """
    labels = LABELS  # 直接使用定义好的标签
 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
 
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(txt_folder, txt_file)
            img_file = txt_file.replace(".txt", ".jpg")  # 假设图像为.png格式
            img_path = os.path.join(img_folder, img_file)
 
            # 检查图像文件是否存在
            if os.path.exists(img_path):
                yolo11_to_labelme(txt_path, img_path, save_folder, labels)
                print(f"已成功转换: {txt_file} -> JSON文件")
            else:
                print(f"图像文件不存在: {img_path}")
 
# 使用示例
txt_folder = r"runs/segment/valpredict/labels"  # YOLO11标签文件夹路径
img_folder = r"data/coco/val2017"  # 图像文件夹路径
save_folder = r"labels_json/val_json"  # JSON文件保存路径
 
convert_yolo11_to_labelme(txt_folder, img_folder, save_folder)