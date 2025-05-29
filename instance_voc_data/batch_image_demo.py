import os
from glob import glob
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-folder',
        action='store_true',
        help='If set, the input is treated as a folder containing images to process in batch.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')

    call_args = vars(parser.parse_args())

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def main():
    init_args, call_args = parse_args()

    inferencer = DetInferencer(**init_args)

    input_path = call_args['inputs']
    output_dir = call_args['out_dir']

    if call_args.pop('batch_folder', False):
        if not os.path.isdir(input_path):
            raise ValueError(f"The input path {input_path} is not a valid directory.")

        image_files = glob(os.path.join(input_path, '*'))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_file in image_files:
            output_file = os.path.join(output_dir,"vis" ,os.path.basename(image_file))
            if os.path.exists(output_file):
                print_log(f'Skipping {image_file} as it is already processed.')
                continue

            print_log(f'Processing {image_file}...')
            call_args['inputs'] = image_file
            inferencer(**call_args)

    else:
        inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args.get('no_save_vis', False)
                                           and call_args.get('no_save_pred', False)):
        print_log(f'results have been saved at {call_args["out_dir"]}')

if __name__ == '__main__':
    # Modify these parameters directly for easier testing
    test_args = {
        'inputs': 'instance_voc_data/out_voc_image',
        'model': 'configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco_voc_yolo.py',
        'weights': "work_dirs/mask-rcnn_r101_fpn_1x_yolo_voc-coco-newid/epoch_12.pth",  # Optional: specify weights if needed
        'out_dir': 'val_vis/out_voc',
        'device': 'cuda:0',
        'pred_score_thr': 0.6,
        'batch_folder': True,
        'palette': 'none'
    }

    # Simulate command-line arguments
    import sys
    sys.argv = [sys.argv[0]] + [
        f'--{k.replace("_", "-")}' if isinstance(v, bool) and v else f'--{k.replace("_", "-")}={v}'
        for k, v in test_args.items() if v is not None
    ]

    main()
