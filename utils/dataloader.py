# --coding:utf-8--
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
from itertools import islice
import torch
import csv
import chardet
from utils.config import config

class DatasetCFP(Dataset):
    def __init__(self, root, mode='train', transform=None, return_path=False):
        """
        参数:
            return_path: 是否返回图像路径(用于CAM可视化)
        """
        self.return_path = return_path
        self.data_list = self.get_files(root, mode)

        if transform is not None:
            self.transforms = transform
        elif mode == 'train':
            self.transforms = T.Compose([
                # 修正：Resize参数顺序（高度→宽度，与config定义一致）
                T.Resize((config.img_height, config.img_weight)),
                T.RandomRotation(30),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((config.img_height, config.img_weight)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def get_files(self, root, mode):
        data_file = os.path.join(root, f'{mode}.csv')
        img_list = []

        try:
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"{data_file} not found!")

            raw_data = open(data_file, 'rb').read()
            encoding = chardet.detect(raw_data)['encoding']
            encodings = [encoding, 'utf-8-sig', 'gbk', 'gb2312']
            for enc in encodings:
                try:
                    with open(data_file, 'r', encoding=enc) as f:
                        csv_reader = csv.reader(f)
                        # 处理CSV头部（自动判断是否跳过）
                        first_line = next(csv_reader, None)
                        if first_line is None:
                            print(f"警告: {data_file} 为空文件")
                            break
                        if not (first_line[0].lower() in ['path', 'image'] or
                                (len(first_line) > 1 and first_line[1].lower() == 'label')):
                            csv_reader = [first_line] + list(csv_reader)
                        for line in csv_reader:
                            try:
                                img_path = os.path.join(root, mode, line[0])
                                if not os.path.exists(img_path):
                                    print(f"警告: 图像不存在 {img_path}")
                                    continue
                                img_list.append([img_path, int(line[1])])
                            except (IndexError, ValueError) as e:
                                print(f"处理行时出错: {line}. 错误: {e}")
                    break  # 编码正确，跳出循环
                except Exception as e:
                    print(f"编码 {enc} 失败，尝试下一个: {e}")
        except Exception as e:
            print(f"读取 {data_file} 时出错: {e}")

        print(f"加载 {mode} 集样本数: {len(img_list)} (来自 {data_file})")
        return img_list

    def __getitem__(self, index):
        image_file, label = self.data_list[index]
        try:
            img = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"错误: 无法打开图像 {image_file} - {e}")
            raise RuntimeError(f"图像加载失败: {image_file}") from e
        img_tensor = self.transforms(img)

        if self.return_path:
            return img_tensor, label, image_file
        else:
            return img_tensor, label

    def __len__(self):
        return len(self.data_list)


def collate_fn(batch):
    """处理带路径和不带路径的不同情况"""
    if len(batch[0]) == 3:  # 包含路径（img_tensor, label, path）
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        paths = [item[2] for item in batch]
        return torch.stack(imgs, 0), labels, paths
    else:  # 不包含路径（img_tensor, label）
        imgs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(imgs, 0), labels

# （可选）主函数仅保留数据加载测试逻辑（验证数据加载是否正常）
if __name__ == "__main__":
    # 测试数据加载功能
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 遍历每个fold测试数据加载
    for k_fold in range(1, 6):
        print(f"\n===== 测试 Fold {k_fold} 数据加载 =====")
        fold_path = f"/public/home/lz_yxy_2706/fenlei/sanfenlei_fg4/data/fold_{k_fold}"

        # 测试训练集加载
        train_csv_path = os.path.join(fold_path, 'train.csv')
        if os.path.exists(train_csv_path):
            train_dataset = DatasetCFP(root=fold_path, mode='train', transform=train_transform)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                pin_memory=True
            )
            print(f"Fold {k_fold} 训练集样本数: {len(train_dataset)}")
            # 测试读取一个batch
            for inputs, labels in train_loader:
                print(f"训练集 Batch 图像形状: {inputs.shape}, 标签数量: {len(labels)}")
                break  # 只测试一个batch
        else:
            print(f"警告: {train_csv_path} 不存在，跳过该fold训练集")
            continue

        # 测试验证集加载
        val_csv_path = os.path.join(fold_path, 'val.csv')
        if os.path.exists(val_csv_path):
            val_dataset = DatasetCFP(root=fold_path, mode='val')
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True
            )
            print(f"Fold {k_fold} 验证集样本数: {len(val_dataset)}")
            # 测试读取一个batch
            for inputs, labels in val_loader:
                print(f"验证集 Batch 图像形状: {inputs.shape}, 标签数量: {len(labels)}")
                break  # 只测试一个batch
        else:
            print(f"警告: {val_csv_path} 不存在，跳过该fold验证集")