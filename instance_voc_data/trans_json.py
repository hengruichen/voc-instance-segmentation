import json

def pretty_format_and_save(input_file, output_file, indent=4, sort_keys=False):
    """
    更美观的JSON格式化保存
    
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param indent: 缩进空格数（推荐2或4）
    :param sort_keys: 是否按键名排序
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用更智能的换行策略
        json_str = json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
            sort_keys=sort_keys,
            separators=(',', ': ')  # 控制冒号后的空格
        )
        
        # 对数组元素进行额外换行处理（使数组更清晰）
        json_str = json_str.replace('[\n        ', '[\n            ')
        json_str = json_str.replace('"\n    ]', '"\n        ]')
        
        f.write(json_str)

# 使用示例（推荐参数）
pretty_format_and_save(
    'output_coco/train/annotations_nofilename.json',
    'data/coco/annotations/train_with_yolo11.json',
    indent=2,          # 缩进2个空格更紧凑
    sort_keys=True     # 键名排序更整齐
)