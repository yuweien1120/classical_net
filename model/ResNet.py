import torch.nn as nn
import torch
from torch.nn import functional as F

class basicblock(nn.Module):
    """
    ResNet18的残差块
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class bottleneck1(nn.Module):
    """
    ResNet50的残差块(stage1)
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 如果使用该1x1卷积块，说明在stage中是第一块卷积块
        if use_1x1conv:
            self.conv1 = nn.Conv2d(input_channels, input_channels,
                                   kernel_size=1)
            self.conv2 = nn.Conv2d(input_channels, input_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1)
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            self.bn1 = nn.BatchNorm2d(input_channels)
            self.bn2 = nn.BatchNorm2d(input_channels)
            self.bn3 = nn.BatchNorm2d(num_channels)
        else:
            self.conv1 = nn.Conv2d(input_channels, int(input_channels/4),
                                   kernel_size=1)
            self.conv2 = nn.Conv2d(int(input_channels/4), int(input_channels/4),
                                   kernel_size=3, padding=1, stride=strides)
            self.conv3 = nn.Conv2d(int(input_channels/4), num_channels,
                                   kernel_size=1)
            self.conv4 = None
            self.bn1 = nn.BatchNorm2d(int(input_channels/4))
            self.bn2 = nn.BatchNorm2d(int(input_channels/4))
            self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)

class bottleneck2(nn.Module):
    """
    ResNet50的残差块(stage2,3,4)
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 如果使用该1x1卷积块，说明在stage中是第一块卷积块，采用下采样
        # 下采样：3x3的卷积核步长为2，则长宽都变为原来的1/2，相应的第三个卷积核的输出通道数是输入通道数的4倍
        if use_1x1conv:
            self.conv1 = nn.Conv2d(input_channels, int(input_channels/2),
                                   kernel_size=1)
            self.conv2 = nn.Conv2d(int(input_channels/2), int(input_channels/2),
                                   kernel_size=3, padding=1, stride=strides)
            self.conv3 = nn.Conv2d(int(input_channels/2), num_channels,
                                   kernel_size=1)
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            self.bn1 = nn.BatchNorm2d(int(input_channels/2))
            self.bn2 = nn.BatchNorm2d(int(input_channels/2))
            self.bn3 = nn.BatchNorm2d(num_channels)
        # 通道数应该先减小再还原
        else:
            self.conv1 = nn.Conv2d(input_channels, int(input_channels/4),
                                   kernel_size=1)
            self.conv2 = nn.Conv2d(int(input_channels/4), int(input_channels/4),
                                   kernel_size=3, padding=1, stride=strides)
            self.conv3 = nn.Conv2d(int(input_channels/4), num_channels,
                                   kernel_size=1)
            self.conv4 = None
            self.bn1 = nn.BatchNorm2d(int(input_channels/4))
            self.bn2 = nn.BatchNorm2d(int(input_channels/4))
            self.bn3 = nn.BatchNorm2d(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)
def resnet_basicblock(input_channels, num_channels, num_residuals,
                      use_1x1conv=False):
    """
    返回一个ResNet18的一个stage
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :param num_residuals: 残差块的个数
    :param use_1x1conv: 使用1x1卷积块下采样
    :return: stage
    """
    blk = []
    # 如果使用1x1卷积块，一定是需要下采样,并且是两个块组合
    if use_1x1conv:
        blk.append(basicblock(input_channels, num_channels,
                              use_1x1conv=True, strides=2))
        blk.append(basicblock(num_channels, num_channels))
        return blk
    # 如果不使用1x1卷积块，则可以多个块一起组合
    else:
        for i in range(num_residuals):
            blk.append(basicblock(num_channels, num_channels))
    return blk

def resnet_bottleneck(input_channels, num_channels, num_residuals,
                      first_stage=False):
    """
    返回一个ResNet50的一个stage
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :param num_residuals: 残差块的个数
    :param first_stage: 第一个stage
    :return: stage
    """
    blk = []
    for i in range(num_residuals):
        if i == 0:
            # 如果是第一个stage的第一个block，不用下采样
            if first_stage:
                blk.append(bottleneck1(input_channels, num_channels, use_1x1conv=True))
            else:
                blk.append(bottleneck2(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            if first_stage:
                blk.append(bottleneck1(num_channels, num_channels))
            else:
                blk.append(bottleneck2(num_channels, num_channels))
    return blk

class ResNet_18(nn.Module):
    """
    定义ResNet18网络结构
    """
    def __init__(self):
        super(ResNet_18, self).__init__()
        self.net1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.net2 = nn.Sequential(*resnet_basicblock(64, 64, 2))
        self.net3 = nn.Sequential(*resnet_basicblock(64, 128, 1, use_1x1conv=True))
        self.net4 = nn.Sequential(*resnet_basicblock(128, 256, 1, use_1x1conv=True))
        self.net5 = nn.Sequential(*resnet_basicblock(256, 512, 1, use_1x1conv=True))
        self.conv = nn.Sequential(self.net1, self.net2, self.net3, self.net4, self.net5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten())
        self.fc = nn.Linear(512, 10)
    def forward(self, X):
        feature = self.conv(X)
        out = self.fc(feature)
        return out

class ResNet_50(nn.Module):
    """
    定义ResNet50网络结构
    """
    def __init__(self):
        super(ResNet_50, self).__init__()
        self.net1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.net2 = nn.Sequential(*resnet_bottleneck(64, 256, 3, first_stage=True))
        self.net3 = nn.Sequential(*resnet_bottleneck(256, 512, 4))
        self.net4 = nn.Sequential(*resnet_bottleneck(512, 1024, 6))
        self.net5 = nn.Sequential(*resnet_bottleneck(1024, 2048, 3))
        self.conv = nn.Sequential(self.net1, self.net2, self.net3, self.net4, self.net5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten())
        self.fc = nn.Linear(2048, 10)
    def forward(self, X):
        feature = self.conv(X)
        out = self.fc(feature)
        return out

if __name__ == '__main__':
    X = torch.rand(1, 3, 32, 32)
    net = ResNet_18()
    print(net(X).shape)
