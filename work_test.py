import os
import re
import glob
import subprocess
import pandas as pd


def get_best_checkpoint(fold_dir):
    # 找到以 best_coco_segm_mAP_epoch_ 开头的文件
    checkpoint_files = glob.glob(os.path.join(fold_dir, 'best_coco_segm_mAP_epoch_*.pth'))
    if not checkpoint_files:
        checkpoint_files = glob.glob(os.path.join(fold_dir, 'epoch_*.pth'))
        # raise FileNotFoundError(f"No checkpoaint files found in {fold_dir}")
    # 假设只有一个符合条件的文件
    return checkpoint_files[0]


def get_script_output(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout


def extract_metrics(output):
    # 使用正则表达式提取 bbox 和 segm 的指标
    bbox_metrics = re.search(r'bbox_mAP_copypaste: ([\d\.]+) ([\d\.]+) ([\d\.]+)', output)
    segm_metrics = re.search(r'segm_mAP_copypaste: ([\d\.]+) ([\d\.]+) ([\d\.]+)', output)

    bbox_results = bbox_metrics.groups() if bbox_metrics else (None, None, None)
    segm_results = segm_metrics.groups() if segm_metrics else (None, None, None)

    return bbox_results, segm_results


def write_to_csv(results, filename='test_results.csv'):
    # 将结果写入DataFrame
    df = pd.DataFrame(results,
                      columns=['Fold', 'bbox_mAP', 'bbox_AP50', 'bbox_AP75', 'segm_mAP', 'segm_AP50', 'segm_AP75'])

    # 将DataFrame写入CSV文件
    df.to_csv(filename, index=False)
    print(f"Results written to {filename}")


# 配置文件路径（替换为你的配置文件路径）
config_file = 'configs/Compared_models/mask-rcnn_r50_MyCoco7.py'
# 工作目录路径（替换为你的工作目录路径）
work_dir = './work_dirs/mask-rcnn_MyCoco7'

# 五折交叉验证
results = []
for fold in range(5):
    fold_dir = os.path.join(work_dir, f'fold_{fold}')
    checkpoint_file = get_best_checkpoint(fold_dir)

    print(f"Processing fold {fold} with checkpoint {checkpoint_file}...")

    # 调用测试脚本并获取输出
    command = f'python tools/test.py {config_file} {checkpoint_file}'
    output = get_script_output(command)

    # 提取指标
    bbox_results, segm_results = extract_metrics(output)

    # 将结果存入列表
    results.append([fold] + list(bbox_results) + list(segm_results))

# 将结果写入CSV
write_to_csv(results)
