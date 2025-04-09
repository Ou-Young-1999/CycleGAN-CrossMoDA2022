from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

# 定义自己的数据集类
class CustomImageDataset(Dataset):
    def __init__(self, data_type, transform=None):
        self.img_dir = 'F:\\CrossMoDA\\CrossMoDA'
        self.data_type = data_type
        self.transform = transform
        self.image_paths_T1 = []
        self.image_paths_T2 = []

        # 将T1和T2模态图片文件路径存储在列表中
        txt_path_T1 = './preprocess/' + self.data_type + 'T1.txt'
        with open(txt_path_T1, 'r') as txt_file:
            for i, row in enumerate(txt_file):
                image_case = os.path.join(self.img_dir, 'T1', row.split('\n')[0])
                for case in os.listdir(image_case):
                    self.image_paths_T1.append(os.path.join(image_case, case))

        txt_path_T2 = './preprocess/' + self.data_type + 'T2.txt'
        with open(txt_path_T2, 'r') as txt_file:
            for i, row in enumerate(txt_file):
                image_case = os.path.join(self.img_dir, 'T2', row.split('\n')[0])
                for case in os.listdir(image_case):
                    self.image_paths_T2.append(os.path.join(image_case, case))

    def __len__(self):
        return max(len(self.image_paths_T1), len(self.image_paths_T2))

    def __getitem__(self, idx):
        img_path_T1 = self.image_paths_T1[idx % len(self.image_paths_T1)] # 在A中取一张照片

        seed = 3407+idx # 设置随机数种子保证可复现性和随机性
        random.seed(seed)
        img_path_T2 = self.image_paths_T2[random.randint(0, len(self.image_paths_T2) - 1)] # 在B中随机取一张

        img_T1 = Image.open(img_path_T1).convert("RGB")  # 确保图像是RGB格式
        img_T2 = Image.open(img_path_T2).convert("RGB")  # 确保图像是RGB格式

        if self.transform:
            img_T1 = self.transform(img_T1)
            img_T2 = self.transform(img_T2)

        return img_T1, img_T2

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

if __name__ == '__main__':
    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 定义图像变换
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     # transforms.RandomCrop((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # 定义数据增强方法
    transform = transforms.Compose([
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 随机垂直翻转
        transforms.RandomVerticalFlip(p=0.5),
        # 随机旋转
        transforms.RandomRotation(degrees=90),
        # 指定尺寸（例如256x256）
        transforms.Resize((256, 256)),
        # 随机调整亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # 将图片转换为Tensor
        transforms.ToTensor(),
        # 归一化到[-1, 1]的范围（针对CycleGAN）
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 创建数据集实例
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    test_dataset = CustomImageDataset(data_type='test', transform=transform)
    print(f'train size: {len(train_dataset)}')
    print(f'test size: {len(test_dataset)}')

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取一个批次的数据
    data_iter = iter(train_loader)
    img_T1, img_T2 = next(data_iter)

    grid_img_T1 = make_grid(img_T1, normalize=True)
    grid_img_T2 = make_grid(img_T2, normalize=True)

    # mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    # std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    # grid_img_T1 = grid_img_T1 * std + mean
    # grid_img_T2 = grid_img_T2 * std + mean

    concat = torch.cat((grid_img_T1, grid_img_T2), dim=1)

    # 可视化网格图片
    plt.imshow(concat.permute(1, 2, 0))  # 调整通道顺序以适应 matplotlib 的要求
    plt.show()