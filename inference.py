import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader  # 导入 DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from itertools import cycle
from sklearn import metrics
from utils.dataloader import DatasetCFP  # 确保正确导入数据加载器
from utils.config import config
from models.des import densenet121  # 或其他模型
import csv
import os

# 绘制混淆矩阵
def plot_confusion_matrix(Y_test, Y_pred, method):
    # 确保保存图像的文件夹存在
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    con_mat = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(con_mat, annot=True, fmt='d', cmap="OrRd", annot_kws={'size': 12, 'color': 'black'}, cbar=False)
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.savefig(os.path.join(results_dir, f'{method}_confusion_matrix.png'))
    plt.close()

# 绘制 ROC 曲线
def plot_roc_curve(label_list_return, predict_return):
    # 绘制 ROC 曲线的代码保持不变
    pass

# 计算评估指标
def compute_metrics(Y_test, Y_pred, n):
    acc = metrics.accuracy_score(Y_test, Y_pred)
    f1 = metrics.f1_score(Y_test, Y_pred, average='macro')
    recall = metrics.recall_score(Y_test, Y_pred, average='macro')
    precision = metrics.precision_score(Y_test, Y_pred, average='macro')
    return acc, f1, recall, precision

def main():
    method = config.model_name  # 获取模型名称（method）

    total_cm = None
    total_acc = []
    total_f1 = []
    total_recall = []
    total_precision = []

    # 遍历每一折
    for k_fold in range(5):
        print(f"\nTesting fold {k_fold}...")

        # 加载模型
        model = densenet121()  # 假设你要使用densenet121模型
        model.eval()
        checkpoint_path = './checkpoints/model/densenet121/model_best.pth.tar'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        # 加载验证集数据
        test_path = f"/public/home/lz_yxy_2706/fenlei/code_fold/data/fold_{k_fold}"
        test_dataset = DatasetCFP(root=test_path, mode='val')
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)  # batch_size 可调

        print(f"Fold {k_fold} Test Samples: {len(test_dataset)}")

        labels = []
        preds = []

        with torch.no_grad():
            for input, target in test_dataloader:
                input = input.cuda()
                target = target.cpu().data.numpy()

                # 预测
                output = model(input)
                smax = nn.Softmax(dim=1)
                output = smax(output)
                pred = np.argmax(output.cpu().data.numpy(), axis=1)

                labels.extend(target)
                preds.extend(pred)

        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        total_cm = cm if total_cm is None else total_cm + cm

        # 计算评估指标
        acc, f1, recall, precision = compute_metrics(labels, preds, config.num_classes)
        total_acc.append(acc)
        total_f1.append(f1)
        total_recall.append(recall)
        total_precision.append(precision)

        print(f"Fold {k_fold} Results:")
        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")
        print(f"Recall (macro): {recall}")
        print(f"Precision (macro): {precision}")

    # 计算平均值
    avg_acc = np.mean(total_acc)
    avg_f1 = np.mean(total_f1)
    avg_recall = np.mean(total_recall)
    avg_precision = np.mean(total_precision)

    print("\nOverall Results:")
    print(f"Confusion Matrix:\n{total_cm}")
    print(f"Average Accuracy: {avg_acc}")
    print(f"Average F1 Score: {avg_f1}")
    print(f"Average Recall (macro): {avg_recall}")
    print(f"Average Precision (macro): {avg_precision}")

    # 可视化混淆矩阵
    plot_confusion_matrix(labels, preds, method)

    # 可视化 ROC 曲线
    label_list_return = np.array(labels)  # 真实标签
    predict_return = np.array(preds)  # 预测标签
    plot_roc_curve(label_list_return, predict_return)

    # 保存评估结果
    header = ['Metrics', 'dAMD', 'nAMD', 'PCV', 'mCNV', 'DR', 'acute VKH', 'RP', 'PIC', 'VRL']
    all_metrics = [total_acc, total_f1, total_recall, total_precision]
    with open(f'./results/{method}_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for metric in all_metrics:
            writer.writerow(metric)


if __name__ == '__main__':
    main()
