import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
from mmengine.structures import InstanceData

# 1. 加载模型和配置
config_file = '/data2/chenhengrui/chenhengrui/RL报告/DL_HW2/mmdetection/configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco_voc_yolo.py'
checkpoint_file = '/data2/chenhengrui/chenhengrui/RL报告/DL_HW2/mmdetection/work_dirs/mask-rcnn_r101_fpn_1x_yolo_voc-coco-newid/epoch_12.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 定义要处理的图像列表
image_ids = ['000002', '000627', '000858', '001564']
base_dir = '/data2/chenhengrui/chenhengrui/RL报告/DL_HW2/mmdetection/data/coco/val2017'

# 3. 获取Proposal Boxes（最终优化版本）
def get_proposals(model, img):
    """使用predict_by_feat方法获取proposal boxes并处理InstanceData"""
    # 手动预处理图像
    img = mmcv.imconvert(img, 'bgr', 'rgb')  # BGR → RGB
    img = np.transpose(img, (2, 0, 1))       # (H, W, C) → (C, H, W)
    img = img[np.newaxis, ...]               # 添加批次维度: (1, C, H, W)
    img = torch.from_numpy(img).float()      # 转为float张量
    
    # 移动到模型所在设备
    if next(model.parameters()).is_cuda:
        img = img.cuda()
    
    # 构建img_metas
    img_metas = [{
        'img_shape': img.shape[2:],           # (height, width)
        'scale_factor': np.array([1.0, 1.0, 1.0, 1.0]),
        'flip': False,
        'ori_shape': img.shape[2:],
        'pad_shape': img.shape[2:],
    }]
    
    # 模型前向传播（仅RPN阶段）
    model.eval()
    with torch.no_grad():
        # 提取特征（避免使用data_preprocessor）
        x = model.extract_feat(img)
        
        # 获取RPN输出
        rpn_outs = model.rpn_head(x)
        
        # 获取proposal配置
        try:
            proposal_cfg = model.test_cfg.rpn
        except:
            proposal_cfg = model.test_cfg
        
        # 生成proposals（使用predict_by_feat）
        proposals = model.rpn_head.predict_by_feat(
            *rpn_outs,
            batch_img_metas=img_metas,
            cfg=proposal_cfg
        )[0]  # 获取第一张图像的proposals
        
        # 处理InstanceData对象
        if isinstance(proposals, InstanceData):
            # 提取边界框和分数
            boxes = proposals.bboxes.cpu().numpy()
            scores = proposals.scores.cpu().numpy()
            # 合并为 [x1, y1, x2, y2, score] 格式
            proposals = np.hstack([boxes, scores.reshape(-1, 1)])
        
        return proposals

# 对每张图像进行处理
for img_id in image_ids:
    img_path = os.path.join(base_dir, f'{img_id}.jpg')
    
    # 检查图像文件是否存在
    if not os.path.exists(img_path):
        print(f"警告: 图像 {img_path} 不存在，跳过处理")
        continue
    
    print(f"\n正在处理图像: {img_path}")
    
    # 创建以图像ID命名的子文件夹
    output_dir = os.path.join('results', img_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建proposal结果的文件夹
    proposal_output_dir = os.path.join('results', 'proposals')
    os.makedirs(proposal_output_dir, exist_ok=True)
    
    # 2. 加载图像
    img = mmcv.imread(img_path)
    img_origin = img.copy()  # 原始图像的副本，用于后续处理
    
    # 获取图像尺寸信息
    height, width = img_origin.shape[:2]
    print(f"图像尺寸: 宽度={width}, 高度={height}")
    
    # 4. 推理获取最终结果
    result = inference_detector(model, img_origin)
    
    # 获取proposal结果
    try:
        proposals = get_proposals(model, img_origin)
        
        # 保存proposal结果到文件
        proposal_file_path = os.path.join(proposal_output_dir, f'{img_id}_proposals.txt')
        with open(proposal_file_path, 'w') as f:
            if len(proposals) > 0:
                for box in proposals:
                    x1, y1, x2, y2, score = box
                    f.write(f'{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}, {score:.4f}\n')
            else:
                f.write("No proposals found.\n")
        print(f"Proposal结果已保存到: {proposal_file_path}")
    except Exception as e:
        print(f"获取proposal失败: {e}")
        continue
    
    # 5. 可视化结果
    plt.figure(figsize=(20, 10))
    
    # 1. 原始图像
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 2. Proposal Boxes - 创建一个新的图像副本用于绘制proposal boxes
    img_for_proposals = img_origin.copy()
    plt.subplot(1, 3, 2)
    plt.title('Stage 1 Proposal Boxes')
    if len(proposals) > 0:
        # 确保proposals是正确的格式
        if proposals.shape[1] >= 5:  # 检查是否有分数列
            # 只显示分数最高的10个proposal
            top_indices = proposals[:, 4].argsort()[::-1][:50]
            top_proposals = proposals[top_indices]
            
            for box in top_proposals:
                x1, y1, x2, y2, score = box
                cv2.rectangle(
                    img_for_proposals, 
                    (round(x1), round(y1)),  # 四舍五入
                    (round(x2), round(y2)),  # 四舍五入
                    (0, 255, 0), 
                    2
                )
                cv2.putText(img_for_proposals, f'{score:.2f}', (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(img_for_proposals, cv2.COLOR_BGR2RGB))
    else:
        plt.title('No Proposals Found')
    
    plt.axis('off')
    
    # 3. 最终检测结果 - 创建一个新的图像副本用于绘制最终检测结果
    img_for_detection = img_origin.copy()
    plt.subplot(1, 3, 3)
    plt.title('Final Detection Results')
    try:
        from mmdet.visualization import DetLocalVisualizer
        visualizer = DetLocalVisualizer()
        visualizer.dataset_meta = model.dataset_meta
        visualizer.add_datasample(
            'result',
            img_for_detection,  # 使用新的图像副本
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, 'detection.jpg'),
            pred_score_thr=0.5
        )
    except:
        model.show_result(img_path, result, score_thr=0.5, 
                          out_file=os.path.join(output_dir, 'detection.jpg'))
    
    # 显示最终结果
    detection_img = mmcv.imread(os.path.join(output_dir, 'detection.jpg'))
    plt.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_results.jpg'))
    plt.close()  # 关闭当前图像，避免内存占用
    
    print(f"图像 {img_id} 的结果已保存到 {output_dir} 目录")

print("\n所有图像处理完成!")

# 可视化所有图像的proposal结果
output_vis_dir = os.path.join('results', "vis")
os.makedirs(output_vis_dir, exist_ok=True)

for img_id in image_ids:
    img_path = os.path.join(base_dir, f'{img_id}.jpg')
    output_proposal_image = os.path.join('results', 'vis', f'{img_id}_proposal.jpg')
    os.makedirs(os.path.dirname(output_proposal_image), exist_ok=True)

    # 检查proposal可视化结果是否已存在
    if os.path.exists(output_proposal_image):
        print(f'跳过 {img_path}，因为其proposal可视化结果已处理。')
        continue

    print(f'处理 {img_path} 的proposal可视化...')

    # 获取proposal结果
    proposals = get_proposals(model, mmcv.imread(img_path))

    # 可视化 proposal
    img = mmcv.imread(img_path)
    img_for_proposals = img.copy()
    if len(proposals) > 0:
        # 只保留分数最高的50个proposal
        if proposals.shape[1] >= 5:  # 检查是否有分数列
            top_indices = proposals[:, 4].argsort()[::-1][:50]
            top_proposals = proposals[top_indices]

            # 绘制 proposal 框
            for box in top_proposals:
                x1, y1, x2, y2, score = box
                cv2.rectangle(
                    img_for_proposals,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),  # 绿色框
                    2
                )
                cv2.putText(
                    img_for_proposals,
                    f'{score:.2f}',
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    # 保存 proposal 可视化结果
    mmcv.imwrite(img_for_proposals, output_proposal_image)
    print(f'Proposal可视化结果已保存到 {output_proposal_image}')

print("\n所有图像的proposal可视化处理完成!")