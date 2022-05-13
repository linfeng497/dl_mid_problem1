import time
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from resnet import ResNet18
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#采用gpu加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#下载数据集
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])])
train_data = torchvision.datasets.CIFAR100("../cifar_data", train=True, transform=transform_train, download=True)
test_data = torchvision.datasets.CIFAR100("../cifar_data", train=False, transform=transform_test, download=True)

#加载数据集
train_iter = DataLoader(train_data, batch_size=64)
test_iter = DataLoader(test_data, batch_size=64)

#CNN框架
net = ResNet18(100)
net = net.to(device)

#损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)

#优化算法
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(),lr = lr, momentum=0.9,weight_decay = 1e-04)

#设置训练参数
train_loss = 0
train_process = 0
test_process = 0
epc = 10

writer = SummaryWriter("../cifar_log")
start_time = time.time()
for i in range(epc):
    net.train()
    for data in train_iter:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        loss = loss_function(y_hat, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_process += 1
        if train_process % 200 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数:{}, Loss:{}".format(train_process, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_process)

    net.eval()
    test_loss, total_accuracy = 0, 0
    with torch.no_grad():
        for data in test_iter:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = loss_function(y_hat, y)
            test_loss += loss.item()
            accuracy = (y_hat.argmax(1) == y).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / 10000))
    writer.add_scalar("test_loss", test_loss, test_process)
    writer.add_scalar("test_accuracy", total_accuracy / 10000, test_process)
    test_process = test_process + 1

    torch.save(net, "base_net_{}.pth".format(i))
    print("模型已保存")

writer.close()
