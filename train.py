import itertools
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
from model import generator
from model import patch_discriminator
from tqdm import tqdm
import os
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化


# 样本的缓存区，提高样本多样性，避免模式崩溃
class ReplayBuffer:
    def __init__(self, max_size=16):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  # 放入一张图像，再从buffer里取一张出来
        to_return = []  # 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  # 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  # 满了就1/2的概率从buffer里取，1/2的概率用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

# 设置学习率为初始学习率乘以给定lr_lambda函数的值
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        # 断言，要让n_epochs > decay_start_epoch 才可以
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  ## return 1-max(0, epoch - decay_start_epoch) / (n_epochs - decay_start_epoch)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


## 每间隔20个迭代就打印图片
def sample_images(batches_done, test_loader, save_dir='./images'):  ## （100/200/300/400...）
    """保存测试集中生成的样本"""
    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    T1, T2 = next(iter(test_loader))  ## 取一张图像
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    real_A = Variable(T1).to(device)   # 取一张真A
    fake_B = G_AB(real_A)  # 用真A生成假B
    real_B = Variable(T2).to(device)   # 取一张真B
    fake_A = G_BA(real_B)  ## 用真B生成假A
    # Arange images along x-axis
    ## make_grid():用于把几个图像按照网格排列的方式绘制出来
    real_A = make_grid(real_A, nrow=batchsize, normalize=True)
    real_B = make_grid(real_B, nrow=batchsize, normalize=True)
    fake_A = make_grid(fake_A, nrow=batchsize, normalize=True)
    fake_B = make_grid(fake_B, nrow=batchsize, normalize=True)
    # Arange images along y-axis
    ## 把以上图像都拼接起来，保存为一张大图片
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_path = os.path.join(save_dir, str(batches_done)+'.png')
    save_image(image_grid, save_path, normalize=False)

# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, num_epochs, save_dir='./models'):
    G_AB, G_BA, D_A, D_B = model
    criterion_GAN, criterion_cycle, criterion_identity = criterion
    optimizer_G, optimizer_D_A, optimizer_D_B = optimizer
    lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B = scheduler
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_id = 0.0
        running_loss_cycle = 0.0
        running_loss_gan = 0.0
        running_loss_da = 0.0
        running_loss_db = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for T1, T2 in t:
                real_A = Variable(T1).to(device)  # 真图像A
                real_B = Variable(T2).to(device)  # 真图像B

                # 全真，全假的标签
                valid = Variable(torch.ones((real_A.size(0), 1, 16, 16)),
                                 requires_grad=False).cuda()  # 定义真的图片label为1 ones((1, 1, 16, 16))
                fake = Variable(torch.zeros((real_A.size(0), 1, 16, 16)),
                                requires_grad=False).cuda()  # 定义假的图片的label为0 zeros((1, 1, 16, 16))

                G_AB.train()
                G_BA.train()
                D_A.train()
                D_B.train()
                # -----------------------
                # 训练生成器
                # 清零梯度
                optimizer_G.zero_grad()

                # 前向传播+计算损失
                # Identity loss（身份损失）：A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                loss_id_B = criterion_identity(G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2

                # GAN loss（对抗损失）
                fake_B = G_AB(real_A)
                # 用B鉴别器鉴别假图像B，训练生成器的目的就是要让鉴别器以为假的是真的，假的太接近真的让鉴别器分辨不出来
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                # 用A鉴别器鉴别假图像A，训练生成器的目的就是要让鉴别器以为假的是真的,假的太接近真的让鉴别器分辨不出来
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Cycle loss（循环一致性损失），之前中realA 通过 A -> B 生成的假图像B，再经过 B -> A ，使得fakeB 得到的循环图像recovA，
                recov_A = G_BA(fake_B)
                loss_cycle_A = criterion_cycle(recov_A, real_A)
                recov_B = G_AB(fake_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = loss_GAN + 10 * loss_cycle + 5 * loss_identity

                # 反向传播和优化
                loss_G.backward()
                optimizer_G.step()

                # -----------------------
                # 判别器 A
                # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
                optimizer_D_A.zero_grad()  # 清零梯度
                loss_real = criterion_GAN(D_A(real_A), valid)
                ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2
                loss_D_A.backward()  # 反向传播
                optimizer_D_A.step()

                # -----------------------
                # 判别器 B
                # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
                optimizer_D_B.zero_grad()  # 清零梯度
                loss_real = criterion_GAN(D_B(real_B), valid)
                ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2
                loss_D_B.backward()  # 将误差反向传播
                optimizer_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2

                # 更新进度条
                running_loss_G += loss_G.item()
                running_loss_D += loss_D.item()
                running_loss_id += loss_identity.item()
                running_loss_cycle += loss_cycle.item()
                running_loss_gan += loss_GAN.item()
                running_loss_da += loss_D_A.item()
                running_loss_db += loss_D_B.item()

                # 每训练20个迭代就保存一组测试集中的图片
                batches_done = epoch * len(train_loader) + t.n
                if batches_done % 20 == 0:
                    sample_images(batches_done, test_loader)

                t.set_postfix(loss_G=running_loss_G / (t.n + 1), loss_D=running_loss_D / (t.n + 1))

        # 记录学习率
        writer.add_scalar('G Learning Rate', optimizer_G.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('D_A Learning Rate', optimizer_D_A.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('D_B Learning Rate', optimizer_D_B.param_groups[0]['lr'], epoch + 1)

        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # 记录损失
        epoch_loss_G = running_loss_G / len(train_loader)
        epoch_loss_D = running_loss_D / len(train_loader)
        epoch_loss_id = running_loss_id / len(train_loader)
        epoch_loss_cycle = running_loss_cycle / len(train_loader)
        epoch_loss_gan = running_loss_gan / len(train_loader)
        epoch_loss_da = running_loss_da / len(train_loader)
        epoch_loss_db = running_loss_db / len(train_loader)

        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_G: {epoch_loss_G:.4f},')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_D: {epoch_loss_D:.4f},')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_id: {epoch_loss_id:.4f},')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_cycle: {epoch_loss_cycle:.4f},')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_gan: {epoch_loss_gan:.4f},')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_da: {epoch_loss_da:.4f},')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss_db: {epoch_loss_db:.4f},')

        writer.add_scalar('Loss_G/train', epoch_loss_G, epoch + 1)
        writer.add_scalar('Loss_D/train', epoch_loss_D, epoch + 1)
        writer.add_scalar('Loss_id/train', epoch_loss_id, epoch + 1)
        writer.add_scalar('Loss_cycle/train', epoch_loss_cycle, epoch + 1)
        writer.add_scalar('Loss_gan/train', epoch_loss_gan, epoch + 1)
        writer.add_scalar('Loss_da/train', epoch_loss_da, epoch + 1)
        writer.add_scalar('Loss_db/train', epoch_loss_db, epoch + 1)

        # 保存模型
        torch.save(G_AB.state_dict(), os.path.join(save_dir, 'epoch_'+str(epoch+1)+'_G_AB.pth'))
        torch.save(G_BA.state_dict(), os.path.join(save_dir, 'epoch_' + str(epoch + 1) + '_G_BA.pth'))
        torch.save(D_A.state_dict(), os.path.join(save_dir, 'epoch_' + str(epoch + 1) + '_D_A.pth'))
        torch.save(D_B.state_dict(), os.path.join(save_dir, 'epoch_' + str(epoch + 1) + '_D_B.pth'))
        tqdm.write(f"epoch: {epoch+1} G_AB, G_BA, D_A, D_B model saved")
        time.sleep(0.5)

if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to train...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 数据预处理
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

    # 加载训练集和验证集
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    test_dataset = CustomImageDataset(data_type='test', transform=transform)
    print(f'Trainset size: {len(train_dataset)}')
    print(f'Testset size: {len(test_dataset)}')

    batchsize = 2
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    print(f'Trainloder size: {len(train_loader)}')
    print(f'Testloder size: {len(test_loader)}')

    # 初始化SummaryWriter
    writer = SummaryWriter('runs/cyclegan')

    # 创建模型
    G_AB = generator(in_channels=3, out_channels=3, channels=64)
    G_BA = generator(in_channels=3, out_channels=3, channels=64)
    D_A = patch_discriminator(in_channels=3, out_channels=1, channels=64)
    D_B = patch_discriminator(in_channels=3, out_channels=1, channels=64)
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)
    model = [G_AB, G_BA, D_A, D_B]

    # 损失函数，MES 二分类的交叉熵，L1loss 相比于L2 Loss保边缘
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss() # 可选
    criterion = [criterion_GAN, criterion_cycle, criterion_identity]

    # 优化器
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0003, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer = [optimizer_G, optimizer_D_A, optimizer_D_B]

    # 设置训练轮次
    num_epochs = 10
    decay_epoch = 5

    # 学习率衰减
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(num_epochs, 0, decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(num_epochs, 0, decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(num_epochs, 0, decay_epoch).step
    )
    scheduler = [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]

    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, num_epochs)

    writer.close()
