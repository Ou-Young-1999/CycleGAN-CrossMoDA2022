# Copyright 2022 Lorna. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import nn
from thop import profile

# 鉴别器（patch级别）
class PatchDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
    ):
        super(PatchDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(channels, int(channels * 2), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 2), int(channels * 4), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 4), int(channels * 8), (4, 4), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 8)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(int(channels * 8), out_channels, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x):
        x = self.main(x)

        return x

# 生成器（ResNet型）
class Generator(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
    ):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, channels, (7, 7), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Downsampling
            nn.Conv2d(channels, int(channels * 2), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(int(channels * 2), int(channels * 4), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4), track_running_stats=True),
            nn.ReLU(True),

            # Residual blocks
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),

            # Upsampling
            nn.ConvTranspose2d(int(channels * 4), int(channels * 2), (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(channels * 2), channels, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, (7, 7), (1, 1), (0, 0)),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.main(x)

        return x


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(_ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
        )

    def forward(self, x):
        identity = x

        x = self.res(x)

        x = torch.add(x, identity)

        return x

# 参数初始化
def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def patch_discriminator(**kwargs):
    model = PatchDiscriminator(**kwargs)
    model.apply(_weights_init)

    return model


def generator(**kwargs):
    model = Generator(**kwargs)
    model.apply(_weights_init)

    return model

if __name__ == '__main__':
    # 定义模型
    G_AB = generator(in_channels=3, out_channels=3, channels=64)
    D_A = patch_discriminator(in_channels=3, out_channels=1, channels=64)

    # 打印结构
    # print(G_AB)
    # print(D_A)

    # 输入张量的形状
    input_tensor = torch.randn(1, 3, 256, 256)

    # 计算 FLOPs 和参数量
    flops, params = profile(G_AB, inputs=(input_tensor,))
    print(f"G_AB FLOPs: {flops / 1e9:.2f} GFLOPs")  # 将 FLOPs 转换为 GFLOPs
    print(f"G_AB Parameters: {params / 1e6:.2f} M")  # 将参数量转换为百万
    flops, params = profile(D_A, inputs=(input_tensor,))
    print(f"D_A FLOPs: {flops / 1e9:.2f} GFLOPs")  # 将 FLOPs 转换为 GFLOPs
    print(f"D_A Parameters: {params / 1e6:.2f} M")  # 将参数量转换为百万

    # 计算输出形状
    output = G_AB(input_tensor)
    print(f"G_AB Output shape: {output.shape}")
    output = D_A(input_tensor)
    print(f"D_A Output shape: {output.shape}")