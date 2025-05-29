# 实例分割实验项目：Mask R-CNN 与 Sparse R-CNN  
本项目基于 mmdetection，比较 Mask R-CNN 和 Sparse R-CNN 在 PASCAL VOC 数据集上的表现。  

## 环境配置  
请先按照 OpenMMLab 官方文档配置环境：👉 [mmdetection 官方安装文档](https://mmdetection.readthedocs.io/en/latest/get_started.html)   

## 目录说明  
`configs/`  
├── `base` # 基础配置文件  
├── `mask_rcnn` # Mask R-CNN 模型配置  
├── `sparse_rcnn` # Sparse R-CNN 模型配置 
`instance_voc_data/` # VOC 数据处理脚本  
`out_voc_image/` # 可视化输出示例  
`parsearg/` # 参数解析脚本  
`batch_image_demo.py` # 批量推理，生成最终预测结果  
`test.py` # 可视化 Mask R-CNN 第一阶段 proposal 框  

## 数据准备  
### 策略一：伪 mask 数据（bounding box 构造）  
1. 下载 PASCAL VOC 2007 和 2012 数据集  
2. 转换为 COCO 格式：`python instance_voc_data/voc2coco.py`  
3. 添加 bbox mask：`python instance_voc_data/addmask.py`  
或2.3步直接用官方脚本：`python tools/dataset_converters/pascal_voc.py`  

### 策略二：YOLOv11x 自动分割 + Labelme 校验  
1. 下载 YOLOv11x 官方权重  
2. 生成实例 label：`python instance_voc_data/create_meta.py`  
3. 创建 labelme json 文件：`python instance_voc_data/create_labelme_json.py`  
4. 在 Labelme 工具中人工校验  
5. 转换为 COCO 格式：`python instance_voc_data/labelme2coco.py`  

## 模型训练  
将 configs/ 中的配置文件放入 mmdetection 的 configs/ 文件夹  
训练命令：`python tools/train.py configs/xxx/your_config.py`  
示例：`python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x.py`  

## 模型测试与可视化  
批量推理：`python batch_image_demo.py`  
配置内容：- 输入图片路径 - 模型权重路径 - 输出保存目录  
可视化 proposal 框：`python test.py`  
配置内容：- 输入图片路径 - 模型权重路径 - 输出保存目录  

## 模型权重与日志  
训练好的模型权重和日志文件：👉 [Google Drive 下载链接](https://drive.google.com/file/d/1Yk5hdC-PHemYEN5FQ0fQ6lSUBj6ZkV5G/view?usp=sharing)
