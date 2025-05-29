_base_ = './mask-rcnn_r50_fpn_1x_yolo11_voc_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
work_dir = './work_dirs/mask-rcnn_r101_fpn_1x_yolo_voc-coco-newid'
