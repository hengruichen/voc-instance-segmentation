_base_ = './sparse-rcnn_r50_fpn_with_mask_1x_coco_yolo.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
work_dir = './work_dirs/sparse-rcnn_r101_fpn_1x_voc-coco_newid_yolo'