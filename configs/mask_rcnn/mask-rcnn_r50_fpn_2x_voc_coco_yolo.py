_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn_voc.py',
    '../_base_/datasets/coco_instance_with_yolovoc.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
visualizer = dict(
    vis_backends=[dict(type='TensorboardVisBackend'),dict(type='LocalVisBackend')],
    type='DetLocalVisualizer'
)
work_dir = './work_dirs/mask-rcnn_r50_fpn_2x_yolo_voc-coco-newid'