import os
import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn import metrics
import csv
import matplotlib.pyplot as plt
from utils.config import config
from utils.dataloader import *
from models.des import densenet121
from models.des169 import densenet169
from models.resnet import *
from models.vgg16 import VGGNet
from models.resnet import resnet18, resnet50
from models.mobilenetv2 import mobilenetv2


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(k_fold):
    # 定义保存检查点的路径，根据当前折数加载对应模型
    checkpoint_path = f'/public/home/lz_yxy_2706/fenlei/BAKD-S/checkpoints/best_model/model_best_fold_{k_fold}.pth.tar'

    # 检查路径是否存在
    if os.path.exists(checkpoint_path):
        # 如果存在，加载模型
        try:
            print(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=True)


            model = mobilenetv2(num_classes=3)
            state_dict = checkpoint['state_dict']
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model.load_state_dict(filtered_state_dict, strict=False)

            # 将模型转到指定设备
            model = model.to(device)
            return model
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {e}")
            return None
    else:
        # 如果不存在，打印错误信息
        print(f"Checkpoint not found at {checkpoint_path}")
        return None


# Function for testing a single model
def test_model(model, test_loader):
    model.eval()
    preds = []
    labels = []
    batch_idx = 0

    with torch.no_grad():
        for input, target in test_loader:
            batch_idx += 1
            input = input.to(device)
            target = target.cpu().numpy()
            labels.extend(target)

            # Predict
            output = model(input)
            smax = nn.Softmax(dim=1)
            output = smax(output)
            pred = np.argmax(output.cpu().numpy(), axis=1)
            preds.extend(pred)

            # Print batch progress
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx}: Processed {len(input)} samples")

    unique_labels = np.unique(labels)
    unique_preds = np.unique(preds)
    print(f"Unique labels in test data: {unique_labels}")
    print(f"Unique predictions: {unique_preds}")

    # Calculate metrics
    cm = metrics.confusion_matrix(labels, preds)
    recall = metrics.recall_score(labels, preds, average='weighted')
    precision = metrics.precision_score(labels, preds, average='weighted')
    f1 = metrics.f1_score(labels, preds, average='weighted')
    kappa = metrics.cohen_kappa_score(labels, preds)

    return cm, recall, precision, f1, kappa


def plot_confusion_matrix(cm, title, file_name):
    try:
        plt.figure(figsize=(10, 7))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=16)
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, fontsize=12)
        plt.yticks(tick_marks, fontsize=12)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], '.2f' if cm.dtype == float else 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=12)

        plt.tight_layout()
        plt.savefig(file_name)
        print(f"Confusion matrix saved to {file_name}")
        plt.close()
    except Exception as e:
        print(f"Error saving confusion matrix to {file_name}: {e}")


if __name__ == "__main__":
    total_cm = None
    fold_cms = []
    total_recall = []
    total_precision = []
    total_f1 = []
    total_kappa = []

    # 创建保存结果的 CSV 文件
    csv_file = 'test_results.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入 CSV 文件的表头，添加标准差列
        writer.writerow(['Fold', 'Recall (weighted)', 'Recall Std', 'Precision (weighted)', 'Precision Std',
                         'F1 Score (weighted)', 'F1 Std', "Cohen's Kappa", "Kappa Std"])

        # Traverse each fold
        for k_fold in range(1, 6):
            print(f"\nTesting fold {k_fold}...")

            # Load the model for the current fold
            model = load_model(k_fold)
            if model is None:
                print(f"Skipping fold {k_fold} due to model loading error.")
                continue

            # Load validation data (changed to val.csv)
            test_path = f"/public/home/lz_yxy_2706/fenlei/BAKD-S/data/fold_{k_fold}"
            test_dataset = DatasetCFP(root=test_path, mode='val')  
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)

            print(f"Fold {k_fold} Test Samples: {len(test_dataset)}")

            # Print first few test samples to confirm loading
            for i in range(min(5, len(test_dataset))):
                img_tensor, label = test_dataset[i]
                print(f"Sample {i}:")
                print(f"  Image Tensor Shape: {img_tensor.shape}")
                print(f"  Label: {label}")

            # Test model and print metrics
            cm, recall, precision, f1, kappa = test_model(model, test_dataloader)

            if total_cm is None:
                total_cm = cm
            else:
                total_cm += cm

            # 存储当前折的混淆矩阵
            fold_cms.append(cm)

            print(f"Fold {k_fold} Confusion Matrix:\n{cm}")
            print(f"Fold {k_fold} Results:")
            print(f"Recall (weighted): {recall}")
            print(f"Precision (weighted): {precision}")
            print(f"F1 Score (weighted): {f1}")
            print(f"Cohen's Kappa: {kappa}\n")

            # 收集每个折的指标值
            total_recall.append(recall)
            total_precision.append(precision)
            total_f1.append(f1)
            total_kappa.append(kappa)

            # 写入当前折的结果到 CSV 文件，暂不计算标准差
            writer.writerow([k_fold, recall, 'N/A', precision, 'N/A', f1, 'N/A', kappa, 'N/A'])

            # 绘制并保存当前折的混淆矩阵
            plot_confusion_matrix(cm, f"Confusion Matrix - Fold {k_fold}", f"confusion_matrix_fold_{k_fold}.png")

    # 计算整体结果和标准差
    recall_mean = np.mean(total_recall)
    recall_std = np.std(total_recall)
    precision_mean = np.mean(total_precision)
    precision_std = np.std(total_precision)
    f1_mean = np.mean(total_f1)
    f1_std = np.std(total_f1)
    kappa_mean = np.mean(total_kappa)
    kappa_std = np.std(total_kappa)

    # 计算平均混淆矩阵
    if fold_cms:
        avg_cm = np.mean(fold_cms, axis=0)
        print("\nAverage Confusion Matrix:")
        print(avg_cm)
    else:
        avg_cm = None
        print("\nNo confusion matrices available to compute average.")

    # Print overall results
    print("\nOverall Results:")
    print(f"Confusion Matrix:\n{total_cm}")
    print(f"Average Recall (weighted): {recall_mean} ± {recall_std}")
    print(f"Average Precision (weighted): {precision_mean} ± {precision_std}")
    print(f"Average F1 Score (weighted): {f1_mean} ± {f1_std}")
    print(f"Average Cohen's Kappa: {kappa_mean} ± {kappa_std}")

    # 将整体结果和平均混淆矩阵结果写入 CSV 文件
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Overall', recall_mean, recall_std, precision_mean, precision_std,
                         f1_mean, f1_std, kappa_mean, kappa_std])
        # 写入平均混淆矩阵信息
        writer.writerow([])  # 空行分隔
        writer.writerow(['Average Confusion Matrix'])
        for row in avg_cm if avg_cm is not None else [[]]:
            writer.writerow(row)

    # 绘制并保存整体的混淆矩阵和平均混淆矩阵
    if total_cm is not None:
        plot_confusion_matrix(total_cm, "Overall Confusion Matrix", "overall_confusion_matrix.png")

    if avg_cm is not None:
        plot_confusion_matrix(avg_cm, "Average Confusion Matrix", "average_confusion_matrix.png")