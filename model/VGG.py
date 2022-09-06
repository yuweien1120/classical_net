import torch.nn as nn

def VGG_block(num_convs, in_channels, out_channels):
    """
    返回一个指定卷积层个数的VGG块
    :param num_convs: 卷积层个数
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :return:指定卷积层个数的VGG块
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, conv_arch):
        super(VGG, self).__init__()
        conv_blks = []
        in_channels = 3
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_blks, nn.Flatten())
        # 使用cifar10作为数据集训练，因此全连接层的神经元个数为500，300，10
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 1 * 1, 500), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(500, 300), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(300, 10)
        )
    def forward(self, X):
        feature = self.conv(X)
        out = self.fc(feature)
        return out