from model.LeNet import LeNet
from model.VGG import VGG

from utils.metric import AverageMeter, accuracy
import os
import random  # to set the python random seed
import numpy  # to set the numpy random seed
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

import wandb
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

def test(model, device, lossfn, test_loader):
    """
    上传模型在测试集上的测试指标到wandb网页，
    测试指标包括：测试标签位于模型输出前1的正确率，测试标签位于模型输出前5的正确率，测试的损失值
    :param model: 网络模型
    :param device: 训练使用的设备，cuda或cpu
    :param lossfn: 损失函数
    :param test_loader: 测试训练集
    :return:
    """
    model.eval()
    test_loss = AverageMeter()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = lossfn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            test_top1.update(prec1.item(), n=target.size(0))
            test_top5.update(prec5.item(), n=target.size(0))
            test_loss.update(loss.item(), n=target.size(0))

    wandb.log({
        "Test top 1 Acc": test_top1.avg,
        "Test top5 Acc": test_top5.avg,
        "Train Loss": test_loss.avg})
    # return test_top1.avg 如果是验证集，则可以返回acc

def train(model, device, train_loader, optimizer, lossfn, epoch, epochs):
    """
    对模型进行一轮训练，并打印和上传相关的训练指标
    训练指标包括：训练标签位于模型输出前1的正确率，训练标签位于模型输出前5的正确率，训练的损失值
    :param model: 网络模型
    :param device: 训练使用的设备，cuda或cpu
    :param train_loader: 训练集
    :param optimizer: 训练优化器
    :param lossfn: 损失函数
    :param epoch: 训练轮数
    :param epochs: 训练总轮数
    :return: 训练标签位于模型输出前1的正确率
    """
    model.train()
    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = lossfn(output, target)

        with torch.no_grad():
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            train_loss.update(loss.item(), n=target.size(0))
            train_top1.update(prec1.item(), n=target.size(0))
            train_top5.update(prec5.item(), n=target.size(0))

        loss.backward()
        optimizer.step()
        # 判断loss当中是不是有元素非空，如果有就终止训练，并打印梯度爆炸
        if np.any(np.isnan(loss.item())):
            print("Gradient Explore")
            break
        # 每训练20个小批量样本就打印一次训练信息
        if batch_idx % 20 == 0:
            print('Epoch: [%d|%d] Step:[%d|%d], LOSS: %.5f' %
                  (epoch, epochs, batch_idx + 1, len(train_loader), loss.item()))

    wandb.log({
        "Train top 1 Acc": train_top1.avg,
        "Train top5 Acc": train_top5.avg,
        "Train Loss": train_loss.avg})
    return train_top1.avg

def main(config):
    # 配置训练模型时使用的设备（cpu/cuda）
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 如果使用cuda则修改线程数和允许加载数据到固定内存
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # 设置随机数种子，保证结果的可复现性
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    numpy.random.seed(config.seed)  # numpy random seed
    # 固定返回的卷积算法，保证结果的一致性
    torch.backends.cudnn.deterministic = True

    # 读取数据集
    transform = transforms.Compose([
        # 将图片尺寸resize到32x32
        transforms.Resize((32, 32)),
        # 将图片转化为Tensor格式
        transforms.ToTensor(),
        # 正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform),
                              batch_size=config.batch_size,
                              shuffle=True, **kwargs)

    test_loader = DataLoader(datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform),
                             batch_size=config.test_batch_size,
                             shuffle=False, **kwargs)
    # 定义使用的网络模型
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))# VGG11
    model = VGG(conv_arch).to(device)
    #model = LeNet()
    #定义损失函数
    lossfn = nn.CrossEntropyLoss().to(device)

    # 定义优化器（梯度下降法）
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    # 追踪模型参数并上传wandb
    wandb.watch(model, log="all")

    # 设置期望正确率
    max_acc = 0
    acc = 0
    # 定义存储模型参数的文件名
    model_path = config.datasets + '_' + config.model + '.pth'

    # 训练模型
    for epoch in range(1, config.epochs + 1):
        acc = train(model, device, train_loader, optimizer, lossfn, epoch, config.epochs)

        # 如果acc为非数，则终止训练
        if np.any(np.isnan(acc)):
            print("NaN")
            break

        # 当训练正确率超出期望值，存储模型参数
        if acc > max_acc:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_path))

        # 每训练完一轮，测试模型的指标
        test(model, device, lossfn, test_loader)

    # 如果正确率非空，则保存模型到wandb
    if not np.any(np.isnan(acc)):
        wandb.save('*.pth')

if __name__ == '__main__':
    # 定义wandb上传项目名
    wandb.init(project="VGG11(CIFAR10)")
    wandb.watch_called = False

    # 定义上传的超参数
    config = wandb.config
    config.datasets = 'CIFAR10'  # 数据集
    config.model = 'VGG11'  # 网络模型
    config.batch_size = 256  # 批量样本数
    config.test_batch_size = 100  # 测试批量样本数
    config.epochs = 10  # 训练轮数
    config.lr = 0.05  # 学习率
    config.momentum = 0.9
    config.no_cuda = False  # 不使用cuda（T/F）
    config.seed = 42  # 随机数种子
    config.log_interval = 20  # 上传数据的间隔（单位：批量数）
    main(config)
