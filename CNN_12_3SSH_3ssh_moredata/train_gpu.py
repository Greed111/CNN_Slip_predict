from model import *
from load_data import *
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

data_x_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/train_x.npy'
data_y_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/train_y.npy'
data_x_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/test_x.npy'
data_y_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/test_y.npy'

train_data = my_dataset(data_x_train, data_y_train, 43)
test_data = my_dataset(data_x_test, data_y_test, 9)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(test_data_size, train_data_size)

# 添加tensorboard
writer = SummaryWriter("logs_train/")

#加载模型和数据集
CNN = my_model().cuda()
train_dataloader = DataLoader(train_data, batch_size=6)
test_dataloader = DataLoader(test_data, batch_size=2)

#损失函数
# loss_fn = nn.CrossEntropyLoss().cuda()       # batch > 10,出来就是巨大低误差
loss_fn = nn.MSELoss().cuda()     # any batch

# 优化器
learning_rate = 0.004
# optimizer = torch.optim.SGD(CNN.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(CNN.parameters(), lr=0.001)


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 30

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    CNN.train()
    for data in train_dataloader:
        samples, targets = data
        samples = samples.cuda()
        targets = targets.cuda()
        outputs = CNN(samples)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 6 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    CNN.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            samples, targets = data
            samples = samples.cuda()
            targets = targets.cuda()
            outputs = CNN(samples)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # accuracy = (outputs.argmax(1) == targets).sum()
            # total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
   # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
   # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

# outputs = outputs.cpu()
# targets = targets.cpu()
writer.close()
torch.save(CNN, "model/sealevel_move_1.pth")
print("模型已保存")