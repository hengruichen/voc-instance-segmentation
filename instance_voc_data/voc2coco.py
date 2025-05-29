import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
import glob
import os
from pathlib import Path

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError(f"Can't find {name} in {root.tag}")
    if length > 0 and len(vars) != length:
        raise ValueError(f"Expect {length} {name} elements, found {len(vars)}")
    return vars[0] if length == 1 else vars

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = get_and_check(root, 'size', 1)
    width = int(get_and_check(size, 'width', 1).text)
    height = int(get_and_check(size, 'height', 1).text)
    
    objects = []
    for obj in root.findall('object'):
        name = get_and_check(obj, 'name', 1).text
        bndbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bndbox, 'xmin', 1).text)
        ymin = int(get_and_check(bndbox, 'ymin', 1).text)
        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax-xmin, ymax-ymin]
        })
    return filename, width, height, objects

def convert_voc_to_coco(voc_root, coco_root, years=['VOC2007','VOC2012']):
    # 创建COCO目录结构
    os.makedirs(os.path.join(coco_root, 'annotations'), exist_ok=True)
    for split in ['train2017', 'val2017']:  # 删除test2017
        os.makedirs(os.path.join(coco_root, split), exist_ok=True)

    # 收集所有数据集划分
    dataset_splits = {
        'train': [],  # 这将包含trainval的数据
        'val': []     # 这将包含2007的test数据
    }

    # 遍历每个VOC年份
    for year in years:
        annotation_dir = os.path.join(voc_root, year, 'Annotations')
        image_dir = os.path.join(voc_root, year, 'JPEGImages')
        
        # 读取trainval.txt作为训练集
        trainval_file = os.path.join(voc_root, year, 'ImageSets', 'Main', 'trainval.txt')
        if os.path.exists(trainval_file):
            with open(trainval_file, 'r') as f:
                for line in f:
                    img_id = line.strip()
                    dataset_splits['train'].append((year, img_id))

        # 只读取2007的test.txt作为验证集
        if year == 'VOC2007':
            test_file = os.path.join(voc_root, year, 'ImageSets', 'Main', 'test.txt')
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    for line in f:
                        img_id = line.strip()
                        dataset_splits['val'].append((year, img_id))

    # 创建类别映射
    categories = {}
    for xml_file in tqdm(glob.glob(os.path.join(voc_root, '**', 'Annotations', '*.xml'), recursive=True),
                        desc="Collecting categories"):
        tree = ET.parse(xml_file)
        for obj in tree.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in categories:
                categories[cls_name] = len(categories) + 1  # COCO类别从1开始

    # 转换数据
    coco_data = {
        'train': {'images': [], 'annotations': [], 'categories': []},
        'val': {'images': [], 'annotations': [], 'categories': []}  # 删除test
    }

    ann_id = 1
    for split in ['train', 'val']:  # 只处理train和val
        for year, img_id in tqdm(dataset_splits[split], desc=f"Processing {split} set"):
            xml_path = os.path.join(voc_root, year, 'Annotations', f'{img_id}.xml')
            img_file = os.path.join(voc_root, year, 'JPEGImages', f'{img_id}.jpg')
            
            if not os.path.exists(xml_path) or not os.path.exists(img_file):
                print(f"Missing data for {img_id}, skipping...")
                continue

            # 解析XML
            filename, width, height, objects = parse_xml(xml_path)
            
            # 复制图片到COCO目录
            dest_dir = os.path.join(coco_root, f'{split}2017')
            shutil.copy(img_file, dest_dir)
            
            # 添加图片信息
            image_id = len(coco_data[split]['images']) + 1
            coco_data[split]['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': width,
                'height': height
            })
            
            # 添加标注信息
            for obj in objects:
                coco_data[split]['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': categories[obj['name']],
                    'bbox': obj['bbox'],
                    'area': obj['bbox'][2] * obj['bbox'][3],
                    'iscrowd': 0,
                    'segmentation': []
                })
                ann_id += 1

        # 添加类别信息
        coco_data[split]['categories'] = [
            {'id': v, 'name': k, 'supercategory': k} 
            for k, v in categories.items()
        ]

        # 保存JSON
        with open(os.path.join(coco_root, 'annotations', f'instances_{split}2017.json'), 'w') as f:
            json.dump(coco_data[split], f)

if __name__ == "__main__":
    # 配置路径
    VOC_ROOT = "data/VOCdevkit"
    COCO_ROOT = "data/coco"
    
    # 转换数据集
    convert_voc_to_coco(
        voc_root=VOC_ROOT,
        coco_root=COCO_ROOT,
        years=['VOC2007', 'VOC2012']
    )