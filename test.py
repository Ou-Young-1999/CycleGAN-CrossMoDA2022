import csv
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import CustomImageDataset
from model import generator
from tqdm import tqdm
import os
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

# 训练函数
def test_model(model, test_loader):
    G_AB, G_BA = model
    # 创建模型保存目录
    if not os.path.exists('result/real_T1'):
        os.makedirs('result/real_T1')
    if not os.path.exists('result/real_T2'):
        os.makedirs('result/real_T2')
    if not os.path.exists('result/fake_T1'):
        os.makedirs('result/fake_T1')
    if not os.path.exists('result/fake_T2'):
        os.makedirs('result/fake_T2')

    G_AB.eval()  # 设置模型为评估模式
    G_BA.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        i = 0
        for T1, T2 in tqdm(test_loader):
            real_A, real_B = T1.to(device), T2.to(device)
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            real_A = 0.5 * real_A + 0.5
            real_B = 0.5 * real_B + 0.5
            fake_A = 0.5 * fake_A + 0.5
            fake_B = 0.5 * fake_B + 0.5

            # 保存图片
            save_image(real_A, 'result/real_T1/'+str(i+1)+'.png', normalize=False)
            save_image(real_B, 'result/real_T2/' + str(i + 1) + '.png', normalize=False)
            save_image(fake_A, 'result/fake_T1/' + str(i + 1) + '.png', normalize=False)
            save_image(fake_B, 'result/fake_T2/' + str(i + 1) + '.png', normalize=False)

            i += 1

    print(f"测试图像翻译完毕")

if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to test...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载测试集
    test_dataset = CustomImageDataset(data_type='test', transform=transform)
    print(f'Testset size: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f'Testloder size: {len(test_loader)}')

    # 导入模型
    print(f'Loading model...')
    G_AB = generator(in_channels=3, out_channels=3, channels=64)
    G_BA = generator(in_channels=3, out_channels=3, channels=64)
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    model = [G_AB, G_BA]
    G_AB.load_state_dict(torch.load('./models/epoch_10_G_AB.pth'))
    G_BA.load_state_dict(torch.load('./models/epoch_10_G_BA.pth'))

    # 训练模型
    test_model(model, test_loader)