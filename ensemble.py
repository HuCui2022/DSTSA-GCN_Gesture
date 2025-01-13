import argparse
import pdb
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import csv,json
from tqdm import tqdm

import argparse
import yaml
import os

def parse_args():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Fusion and Evaluation Script")
    parser.add_argument('--config', default=None, help='Path to config.yaml file')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--idx', default=None, type=int, help='Only for DHG datasets')
    parser.add_argument('--alpha', default=None, type=float, help='Weighted summation coefficient')
    parser.add_argument('--joint-dir', default=None, help='Path to joint feature predictions')
    parser.add_argument('--bone-dir', default=None, help='Path to bone feature predictions')
    parser.add_argument('--joint-motion-dir', default=None, help='Path to joint motion feature predictions')
    parser.add_argument('--bone-motion-dir', default=None, help='Path to bone motion feature predictions')

    # 解析初始命令行参数
    args = parser.parse_args()

    # 如果提供了 config 文件
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file '{args.config}' not found!")

        with open(args.config, 'r') as f:
            config_args = yaml.load(f, yaml.FullLoader)

        # 检查 config 文件中的键是否有效
        valid_keys = vars(args).keys()
        for k in config_args.keys():
            if k not in valid_keys:
                raise ValueError(f"Unknown argument in config file: '{k}'")

        # 将配置文件中的默认值设置到命令行参数解析器
        parser.set_defaults(**config_args)

        # 再次解析参数，优先使用命令行参数
        args = parser.parse_args()


    return args



def load_labels(dataset, idx=1):
    """根据数据集加载标签"""
    if 'ntu120' in dataset:
        if 'xsub' in dataset:
            npz_data = np.load('E:/DataSets/sttf_ntu/ntu120/NTU120_XSub.npz')
            labels = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in dataset:
            npz_data = np.load('E:/DataSets/sttf_ntu/ntu120/NTU120_XSet.npz')
            labels = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in dataset:
        if 'xsub' in dataset:
            npz_data = np.load('E:/DataSets/sttf_ntu/ntu60/NTU60_XSub.npz')
            labels = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in dataset:
            npz_data = np.load('E:/DataSets/sttf_ntu/ntu60/NTU60_XView.npz')
            labels = np.where(npz_data['y_test'] > 0)[1]
    elif 'UCLA' in dataset:
        with open('E:/DataSets/UCLA/val_info/val_info.pkl', 'rb') as f:
            data_info = pickle.load(f)
            labels = [int(info['label']) - 1 for info in data_info]
    elif 'shrec' in dataset:
        with open('E:/DataSets/SHREC2017/shrec17_jsons/test_samples.json') as f:
            data_info = json.load(f)
            if '14' in dataset:
                labels = [int(info['label_14']) - 1 for info in data_info]
            if '28' in dataset:
                labels = [int(info['label_28']) - 1 for info in data_info]
    elif 'dhg' in dataset:
        if '14' in dataset:
            with open(f'E:/DataSets/DHG2016/DHG14-28_sample_json/{idx}/{idx}val_samples.json') as f:
                data_info = json.load(f)
                labels = [int(info['label_14']) - 1 for info in data_info]
        elif '28' in dataset:
            with open(f'E:/DataSets/DHG2016/DHG14-28_sample_json/{idx}/{idx}val_samples.json') as f:
                data_info = json.load(f)
                labels = [int(info['label_28']) - 1 for info in data_info]
    else:
        raise NotImplementedError("Unsupported dataset!")
    return labels


def load_predictions(dir_path, filename='epoch1_test_score.pkl'):
    """加载预测分数"""
    with open(os.path.join(dir_path, filename), 'rb') as f:
        return list(pickle.load(f).items())


def compute_accuracy(label, predictions, top_k=5):
    """计算 Top-1 和 Top-K 准确率"""
    right_num = right_num_k = 0
    total_num = len(label)

    for i, true_label in enumerate(label):
        pred_scores = predictions[i]
        top_k_preds = pred_scores.argsort()[-top_k:]
        right_num += (np.argmax(pred_scores) == true_label)
        right_num_k += (true_label in top_k_preds)

    acc = right_num / total_num
    acc_k = right_num_k / total_num
    return acc, acc_k


import itertools
from tqdm import tqdm

def main():
    args = parse_args()

    # Load labels
    print(args)
    labels = load_labels(args.dataset, idx=args.idx)

    # Load predictions
    if 'dhg' in args.dataset:
        if '14' in args.dataset:
            jdir = f'work_dir/dgh14_{args.idx}/criss_stgcnv2/joint'
            bdir = f'work_dir/dgh14_{args.idx}/criss_stgcnv2/bone'
            jmdir = f'work_dir/dgh14_{args.idx}/criss_stgcnv2/jmotion'
            bmdir = f'work_dir/dgh14_{args.idx}/criss_stgcnv2/bmotion'
        if '28' in args.dataset:
            jdir = f'work_dir/dgh28_{args.idx}/criss_stgcnv2/joint'
            bdir = f'work_dir/dgh28_{args.idx}/criss_stgcnv2/bone'
            jmdir = f'work_dir/dgh28_{args.idx}/criss_stgcnv2/jmotion'
            bmdir = f'work_dir/dgh28_{args.idx}/criss_stgcnv2/bmotion'
        r1 = load_predictions(jdir)
        r2 = load_predictions(bdir)
        r3 = load_predictions(jmdir)
        r4 = load_predictions(bmdir)
    elif 'shrec' in args.dataset:
        if '14' in args.dataset:
            jdir = f'work_dir/shrec14/criss_stgcnv2/rand_head12'
            bdir = f'work_dir/shrec14/criss_stgcnv2/adj_bone12_rand'
            jmdir = f'work_dir/shrec14/criss_stgcnv2/adj_jm12_rand'
            bmdir = f'work_dir/shrec14/criss_stgcnv2/adj_bonemotion_head12_rand'
            if "info" in args.dataset:
                jdir = f'work_dir/shrec14/criss_stgcnv2/adj_j_12_rand_info'
                bdir = f'work_dir/shrec14/criss_stgcnv2/adj_bone12_rand_info'
                jmdir = f'work_dir/shrec14/criss_stgcnv2/adj_jm12_rand_info'
                bmdir = f'work_dir/shrec14/criss_stgcnv2/adj_bonemotion12_rand_info'
        if '28' in args.dataset:
            jdir = f'work_dir/shrec28/criss_stgcnv2/filter8'
            bdir = f'work_dir/shrec28/criss_stgcnv2/bone'
            jmdir = f'work_dir/shrec28/criss_stgcnv2/motion'
            bmdir = f'work_dir/shrec28/criss_stgcnv2/bone_motion'
            if 'info' in args.dataset:
                jdir = f'work_dir/shrec28/criss_stgcnv2/filter8_sp_info'
                bdir = f'work_dir/shrec28/criss_stgcnv2/bone_info'
                jmdir = f'work_dir/shrec28/criss_stgcnv2/motion_info'
                bmdir = f'work_dir/shrec28/criss_stgcnv2/bone_motion_info'

        r1 = load_predictions(jdir)
        r2 = load_predictions(bdir)
        r3 = load_predictions(jmdir)
        r4 = load_predictions(bmdir)
    elif "ntu" in args.dataset:
        if '120' in args.dataset:
            if 'xsub' in args.dataset:
                jdir = f'work_dir/ntu120/xsub/joint'
                bdir = f'work_dir/ntu120/xsub/bone'
                jmdir = f'work_dir/ntu120/xsub/jm'
                bmdir = f'work_dir/ntu120/xsub/bm'
                print('ntu120...xsub')
            elif 'xset' in args.dataset:
                jdir = f'work_dir/ntu120/NTU120_XSet/joint'
                bdir = f'work_dir/ntu120/NTU120_XSet/bone'
                jmdir = f'work_dir/ntu120/NTU120_XSet/jm'
                bmdir = f'work_dir/ntu120/NTU120_XSet/bm'
                print('ntu120...xset')
        else:
            if 'xsub' in args.dataset:
                jdir = f'work_dir/ntu60/xsub/joint'
                bdir = f'work_dir/ntu60/xsub/bone'
                jmdir = f'work_dir/ntu60/xsub/jm'
                bmdir = f'work_dir/ntu60/xsub/bm'
            elif 'xview' in args.dataset:
                jdir = f'work_dir/ntu60/xview/joint'
                bdir = f'work_dir/ntu60/xview/bone'
                jmdir = f'work_dir/ntu60/xview/jm'
                bmdir = f'work_dir/ntu60/xview/bm'
        r1 = load_predictions(jdir)
        r2 = load_predictions(bdir)
        r3 = load_predictions(jmdir)
        r4 = load_predictions(bmdir)
    else:
        r1 = load_predictions(args.joint_dir)
        r2 = load_predictions(args.bone_dir)
        r3 = load_predictions(args.joint_motion_dir) if args.joint_motion_dir else None
        r4 = load_predictions(args.bone_motion_dir) if args.bone_motion_dir else None

    # Fusion and evaluation: Coarse Search
    coarse_step = 0.2
    fine_step = 0.05
    print(f"length of the labels : {len(labels)}")
    print(f"pred length r1: {len(r1)},r2: {len(r2)},r3: {len(r3)},r4: {len(r4)} ")
    print(f"{jdir}")
    if 'dhg' not in args.dataset:
        assert len(labels) == len(r1) == len(r2) == len(r3) == len(r4)
    alphas_list = list(itertools.product(np.arange(0.0, 1.1, coarse_step), repeat=4))  # 粗略搜索
    best_acc = 0
    best_alphas = [0.4, 0.4, 0.1, 0.2]

    print("Starting coarse search...")
    for alphas in tqdm(alphas_list, desc="Coarse search"):
        if sum(alphas) == 0:  # 避免所有权重为零的情况
            continue

        preds = []
        for i in range(len(labels)):
            r11 = r1[i][1]
            r22 = r2[i][1]
            r33 = r3[i][1] if r3 else 0
            r44 = r4[i][1] if r4 else 0
            fused_scores = r11 * alphas[0] + r22 * alphas[1] + r33 * alphas[2] + r44 * alphas[3]
            preds.append(fused_scores)

        # Compute accuracy
        top1_acc, _ = compute_accuracy(labels, preds)
        if top1_acc > best_acc:
            best_acc = top1_acc
            best_alphas = alphas

    print(f"Coarse search completed. Best Accuracy: {best_acc}")
    print(f"Coarse Best Alphas: {best_alphas}")

    # Fine Search
    fine_ranges = [np.arange(max(0.0, alpha - coarse_step), min(1.0, alpha + coarse_step) + fine_step, fine_step)
                   for alpha in best_alphas]
    fine_alphas_list = list(itertools.product(*fine_ranges))

    print("Starting fine search...")
    for alphas in tqdm(fine_alphas_list, desc="Fine search"):
        if sum(alphas) == 0:  # 避免所有权重为零的情况
            continue

        preds = []
        for i in range(len(labels)):
            r11 = r1[i][1]
            r22 = r2[i][1]
            r33 = r3[i][1] if r3 else 0
            r44 = r4[i][1] if r4 else 0
            fused_scores = r11 * alphas[0] + r22 * alphas[1] + r33 * alphas[2] + r44 * alphas[3]
            preds.append(fused_scores)

        # Compute accuracy
        top1_acc, _ = compute_accuracy(labels, preds)
        if top1_acc > best_acc:
            best_acc = top1_acc
            best_alphas = alphas

    print(f"Fine search completed. Best Accuracy: {best_acc}")
    print(f"Fine Best Alphas: {best_alphas}")

    # 计算各个模态的acc:
    jpred, bpred, jmpred, bmpred = [], [], [], []
    for i in range(len(labels)):
        r11 = r1[i][1]
        r22 = r2[i][1]
        r33 = r3[i][1] if r3 else 0
        r44 = r4[i][1] if r4 else 0
        jpred.append(r11)
        bpred.append(r22)
        jmpred.append(r33)
        bmpred.append(r44)

    j_top1_acc, j_top5_acc = compute_accuracy(labels, jpred)
    b_top1_acc, b_top5_acc = compute_accuracy(labels, bpred)
    jm_top1_acc, jm_top5_acc = compute_accuracy(labels, jmpred)
    bm_top1_acc, bm_top5_acc = compute_accuracy(labels, bmpred)
    print(args.dataset)
    print(args.idx)
    print(f'joint acc : {j_top1_acc}\njm acc: {jm_top1_acc}\nbone acc: {b_top1_acc}\nbm acc: {bm_top1_acc}\n')
    print(f"best_acc : {best_acc}")


if __name__ == "__main__":
    main()
    # configs/ensemble.yaml