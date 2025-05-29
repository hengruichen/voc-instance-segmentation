import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
data_root = "/data2/chenhengrui/chenhengrui/RL报告/DL_HW2/mmdetection/instance_voc_data"
save_dir = "instance_voc_data/parsearg"
os.makedirs(save_dir, exist_ok=True)

# 初始化数据结构
data_groups = {
    "mask": {"bbox": {}, "segm": {}},
    "sparse": {"bbox": {}, "segm": {}}
}

# 遍历文件夹收集数据
for folder in os.listdir(data_root):
    if not os.path.isdir(os.path.join(data_root, folder)):
        continue

    # 分类处理mask和sparse实验
    if folder.startswith("mask"):
        group = "mask"
    elif folder.startswith("sparse"):
        group = "sparse"
    else:
        continue

    # 读取bbox数据 (vis_data.csv)
    bbox_path = os.path.join(data_root, folder, "vis_data.csv")
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            epochs, values = [], []
            for row in reader:
                epochs.append(int(row[1]))  # 第二列是epoch
                values.append(float(row[2]))  # 第三列是mAP
            data_groups[group]["bbox"][folder] = (epochs, values)

    # 读取segm数据 (vis_data(1).csv)
    segm_path = os.path.join(data_root, folder, "vis_data (1).csv")
    if os.path.exists(segm_path):
        with open(segm_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            epochs, values = [], []
            for row in reader:
                epochs.append(int(row[1]))
                values.append(float(row[2]))
            data_groups[group]["segm"][folder] = (epochs, values)

# 设置专业美观的绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 创建画布与子图
fig, axs = plt.subplots(2, 2, figsize=(18, 10), dpi=120)
plt.suptitle("(Depth vs Epoch)", y=1.02, fontsize=14, fontweight='bold')

# 定义专业配色方案
palette = {
    'Exp 1': '#4C72B0',  # 深蓝
    'Exp 2': '#DD8452',  # 橙红
    'Exp 3': '#55A868'   # 翠绿
}

# 定义颜色名字列表
color_labels = ["r50+12epochs", "r50+24epochs", "r101+12epochs"]

# 绘图函数增强版
def plot_enhanced(ax, data, title):
    for idx, (exp_label, (epochs, values)) in enumerate(data.items()):
        # 使用颜色名字列表中的名字
        label = color_labels[idx % len(color_labels)]  # 循环分配名字
        ax.plot(epochs, values,
                lw=2.5,
                alpha=0.8,
                label=label,  # 使用颜色名字
                marker='o',         # 添加数据点标记
                markersize=4,
                markeredgecolor='w')  # 白色描边

    # 专业图表装饰
    ax.set_title(title, pad=12, fontsize=12, fontweight='semibold')
    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("mAP", labelpad=8)
    ax.grid(True, alpha=0.3)

    # 添加半透明背景色
    ax.set_facecolor('#F5F5F5')  # 浅灰背景

    # 美化图例
    ax.legend(frameon=True,
              facecolor='white',
              edgecolor='#DDDDDD',
              loc='lower right')

# 应用样式到所有子图
plot_enhanced(axs[0, 0], data_groups["mask"]["bbox"], "Mask R-CNN - BBox mAP")
plot_enhanced(axs[0, 1], data_groups["mask"]["segm"], "Mask R-CNN - Segm mAP")
plot_enhanced(axs[1, 0], data_groups["sparse"]["bbox"], "Sparse R-CNN - BBox mAP")
plot_enhanced(axs[1, 1], data_groups["sparse"]["segm"], "Sparse R-CNN - Segm mAP")

# 整体调整
plt.tight_layout(pad=2.5)
plt.savefig(os.path.join(save_dir, 'enhanced_performance_plot.png'),
            bbox_inches='tight',
            facecolor='white')  # 纯白背景
plt.close()

print(f"所有图表已保存至：{os.path.join(save_dir, 'enhanced_performance_plot.png')}")