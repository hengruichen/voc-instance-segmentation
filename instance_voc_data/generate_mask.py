import os
import cv2
import numpy as np
from pycocotools.coco import COCO

# ====== 配置 ======
ann_file = 'data/coco/annotations/train_with_yolo11_class.json'
img_dir = 'data/coco/train2017'
output_vis_dir = 'yolooutput_instance_visualization'   # 黑色背景可视化
output_overlay_dir = 'yolooutput_overlay'              # 原图叠加可视化
os.makedirs(output_vis_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

# 加载 COCO
coco = COCO(ann_file)
cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)
cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

# 随机颜色（每个实例独立）
np.random.seed(42)
instance_colors = {}

# 遍历图片
for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_filename = img_info['file_name']
    width, height = img_info['width'], img_info['height']
    img_path = os.path.join(img_dir, img_filename)

    if not os.path.exists(img_path):
        print(f"Warning: {img_filename} not found, skipping.")
        continue

    img_black = np.zeros((height, width, 3), dtype=np.uint8)  # 黑色背景图
    img_overlay = cv2.imread(img_path)                        # 原图叠加图

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        mask = coco.annToMask(ann)
        ann_id = ann['id']

        # 分配随机颜色（每个实例独立）
        if ann_id not in instance_colors:
            instance_colors[ann_id] = np.random.randint(0, 255, 3)
        color = instance_colors[ann_id].tolist()

        # ===== 黑背景图填充颜色 =====
        img_black[mask == 1] = color

        # ===== 原图上叠加半透明颜色 =====
        img_overlay[mask == 1] = img_overlay[mask == 1] * 0.5 + np.array(color) * 0.5

        # ===== 绘制轮廓（黑背景 + 原图都画） =====
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_black, contours, -1, (255, 255, 255), thickness=2)   # 白色轮廓
        cv2.drawContours(img_overlay, contours, -1, (255, 255, 255), thickness=2) # 白色轮廓

    # 保存黑背景可视化
    out_path_black = os.path.join(output_vis_dir, f"{img_filename.split('.')[0]}_instance_vis.png")
    cv2.imwrite(out_path_black, img_black)

    # 保存原图叠加可视化
    out_path_overlay = os.path.join(output_overlay_dir, f"{img_filename.split('.')[0]}_overlay.png")
    cv2.imwrite(out_path_overlay, img_overlay)

    print(f"Saved: {out_path_black}")
    print(f"Saved: {out_path_overlay}")

print("All visualizations saved!")
