import numpy as np
from load_data import *
import torch
from model import *
from torch.utils.tensorboard import SummaryWriter

def single_pr(x, y, m):
    x = saved_model(torch.tensor(x, device='cuda'))
    x = x.cpu().detach().numpy()
    print(m, x.shape, y.shape)
    np.save('datasave/y.npy', y)
    np.save('datasave/x.npy', x)


saved_model = torch.load('model/sealevel_move0.pth')
data_x_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/train_x.npy'
data_y_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/train_y.npy'
data_x_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/test_x.npy'
data_y_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/test_y.npy'
data_x_val = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/otherdata/test_x.npy'
data_y_val = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/otherdata/test_y.npy'

train_data = my_dataset(data_x_train, data_y_train, 43)
test_data = my_dataset(data_x_test, data_y_test, 9)


x_ori = np.zeros((12, 72, 31, 156), dtype=np.float32)
x_now = np.zeros((12, 72, 31, 156), dtype=np.float32)
y_num = np.zeros((12, 6, 31, 156), dtype=np.float32)
time = 80       # 预测时间点

writer = SummaryWriter("logs_MOVE")
# loss_func = nn.MSELoss()

predict_data_x, truth_data_y = test_data.predict_ssh(time)
x_ori[0, :, :, :] = predict_data_x
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()
# truth_data_y = np.squeeze(truth_data_y)
y_num[0, :, :, :] = predict_data_x

for i in [-1, -2, -3, -4, -5, -6]:
      x_now[0, i, :, :] = y_num[0, i, :, :]
for i in range(66):
      x_now[0, i, :, :] = x_ori[0, i+6, :, :]

for j in range(1, 12):
      x_ori[j, :, :, :] = x_now[j-1, :, :, :]
      predict_data_x = x_ori[j, :, :, :].reshape(1, 72, 31, 156)
      predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
      predict_data_x = predict_data_x.cpu().detach().numpy()
      y_num[j, :, :, :] = np.squeeze(predict_data_x)
      for i in [-1, -2, -3, -4, -5, -6]:
            x_now[j, i, :, :] = y_num[j, i, :, :]
      for i in range(66):
            x_now[j, i, :, :] = x_ori[j, i + 6, :, :]

      # 预测时间段,注意x和y形状

predict_data_x, truth_data_y, piece = train_data.predict_ssh_area(time, time+11)
print(truth_data_y.shape, y_num.shape)
np.save('datasave/y_12_6.npy', truth_data_y)
np.save('datasave/x_12_6.npy', y_num)