import os
import random
import time
import torch
import numpy as np
import warnings
import shutil
import logging
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.config import config
from utils.dataloader import *
from utils.utils import *
from utils.progress_bar import *
from models.des import densenet121
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models.mobilenetv2 import mobilenetv2

# 配置日志
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_least_used_gpu():
    gpus = list(range(torch.cuda.device_count()))
    free_mem = [torch.cuda.mem_get_info(i)[0] for i in gpus]
    return free_mem.index(max(free_mem))  # 返回空闲显存最大的 GPU ID


os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机种子初始化
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# 工具函数：确保目录存在
def ensure_dir_exists(dir_path):
    """
    确保目录存在；如果不存在，则创建。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 保存模型检查点
def save_checkpoint(state, epoch, save_dir, k_fold):
    """
    保存训练模型的检查点
    """
    ensure_dir_exists(save_dir)
    filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth.tar")
    torch.save(state, filename)


# 训练主函数
def train(k):
    fold = config.model_name

    # 创建必要目录
    ensure_dir_exists(os.path.join(config.weights, "model", str(fold)))
    ensure_dir_exists(os.path.join(config.best_models, str(fold)))
    ensure_dir_exists(os.path.join(config.logs, str(fold)))
    ensure_dir_exists(os.path.join(config.runs, str(fold)))

    # 教师模型与学生模型
    teacher_model = densenet121().to(device)
    student_model = mobilenetv2(num_classes=3).to(device)
    student_pretrained_path = '/public/home/lz_yxy_2706/fenlei/BAKD-S/models/mobilenetv2_1.0-0c6065bc.pth'

    teacher_pretrained_loaded = False
    try:
        teacher_model.load_pretrained_weights()
        teacher_pretrained_loaded = True
        logging.info(f"成功加载教师模型预训练权重")
    except Exception as e:
        logging.error(f"加载教师模型预训练权重失败: {e}")

    student_pretrained_loaded = False
    try:
        student_model.load_pretrained_weights(
            '/public/home/lz_yxy_2706/fenlei/BAKD-S/models/mobilenetv2_1.0-0c6065bc.pth')
        student_pretrained_loaded = True
        logging.info(f"成功加载学生模型预训练权重")
    except Exception as e:
        logging.error(f"加载学生模型预训练权重失败: {e}")

    # 加载教师模型对应折的最佳权重
    teacher_best_model_path = f'/public/home/lz_yxy_2706/fenlei/BAKD-S/checkpoints-T/best_model/model_best_fold_{k}.pth.tar'
    try:
        teacher_checkpoint = torch.load(teacher_best_model_path)
        teacher_model.load_state_dict(teacher_checkpoint["state_dict"])
        logging.info(f"成功加载教师模型第 {k} 折的最佳权重")
    except Exception as e:
        logging.error(f"加载教师模型第 {k} 折的最佳权重失败: {e}")

    # 调整学习率
    lr = 0.0005
    optimizer = optim.Adam(student_model.parameters(), lr=lr, amsgrad=True, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    # 调整学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 恢复训练
    start_epoch = 0
    resume = False
    if resume:
        checkpoint_path = os.path.join(config.best_models, str(fold), "model_best.pth.tar")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        student_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 数据加载
    path = f'/public/home/lz_yxy_2706/fenlei/BAKD-S/data/fold_{k}'
    train_dataloader = DataLoader(DatasetCFP(root=path, mode='train'), batch_size=config.batch_size, shuffle=True,
                                  pin_memory=True)

    best_acc = 0  # 用于记录最佳准确率
    best_model_path = ""  # 用于记录最佳模型路径
    train_losses_list = []  # 用于记录每个 epoch 的训练损失

    # 调整蒸馏温度和蒸馏损失权重
    temperature = 4.0
    alpha = 0.7

    # 训练循环
    for epoch in range(start_epoch, config.epochs):
        scheduler.step()
        train_losses = AverageMeter()
        train_top1 = AverageMeter()

        teacher_model.eval()
        student_model.train()
        for iter, (input, target) in enumerate(train_dataloader):
            input = input.to(device)
            target = target.long().to(device)


            # 教师模型输出
            with torch.no_grad():
                teacher_output = teacher_model(input)

                teacher_logits = teacher_output / temperature
                teacher_probs = F.softmax(teacher_logits, dim=1)

            # 学生模型输出
            student_output = student_model(input)
            student_logits = student_output / temperature
            student_probs = F.log_softmax(student_logits, dim=1)

            # 蒸馏损失
            distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

            # 交叉熵损失
            ce_loss = criterion(student_output, target)

            # 总损失
            loss = alpha * distillation_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            precision1, _ = accuracy(student_output, target, topk=(1, 2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1[0], input.size(0))

        epoch_loss = train_losses.avg
        train_losses_list.append(epoch_loss)
        logging.info(f"Epoch {epoch}, Train Loss: {epoch_loss}, Train Acc: {train_top1.avg}")
        print(f"Epoch {epoch}, Train Loss: {epoch_loss}, Train Acc: {train_top1.avg}")

        # 假设这里的train_top1.avg是当前epoch的准确率，与最佳准确率比较
        if train_top1.avg > best_acc:
            best_acc = train_top1.avg
            # 新的最佳模型保存目录
            save_best_model_dir = '/public/home/lz_yxy_2706/fenlei/BAKD-S/checkpoints/best_model'
            ensure_dir_exists(save_best_model_dir)
            best_model_filename = os.path.join(save_best_model_dir, f"model_best_fold_{k}.pth.tar")
            state = {
                "epoch": epoch + 1,
                "model_name": "mobilenetv2",
                "state_dict": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "fold": fold
            }
            torch.save(state, best_model_filename)
            best_model_path = best_model_filename

    # 保存最佳模型路径到指定路径
    save_best_model_path_dir = '/public/home/lz_yxy_2706/fenlei/BAKD-S/checkpoints/best_model'
    ensure_dir_exists(save_best_model_path_dir)
    with open(os.path.join(save_best_model_path_dir, f"best_model_path_fold_{k}.txt"), 'w') as f:
        f.write(best_model_path)

    # 保存模型
    save_dir = os.path.join(config.weights, "model", str(fold))
    save_checkpoint({
        "epoch": epoch + 1,
        "model_name": "mobilenetv2",
        "state_dict": student_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fold": fold
    }, epoch, save_dir, k)

    # 绘制损失曲线
    plt.plot(range(start_epoch, config.epochs), train_losses_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Curve for Fold {k}')
    plt.savefig(f'training_loss_curve_fold_{k}.png')
    plt.close()


start = time.time()

if __name__ == "__main__":
    for k_fold_test in range(1, 6):
        print(f"\nFold {k_fold_test}")
        writer = SummaryWriter(f"./runs/{config.model_name}")
        train(k_fold_test)

print(f"Total time: {time.time() - start:.2f} seconds")