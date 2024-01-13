import numpy as np
from load_data import *
import torch
from model import *


# 目前最好的模型是

def single_pr(x, y, m):
    x = saved_model(torch.tensor(x, device='cuda'))
    x = x.cpu().detach().numpy()
    print(m, x.shape, y.shape)
    np.save('datasave/y.npy', y)
    np.save('datasave/x.npy', x)

saved_model = torch.load('model/sealevel_move0.pth')
data_x_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/train_x.npy'
data_y_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/train_y.npy'
data_x_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/test_x.npy'
data_y_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/test_y.npy'

train_data = my_dataset(data_x_train, data_y_train, 43)
test_data = my_dataset(data_x_test, data_y_test, 9)

# time = 472         # 预测时间点
# predict_data_x, truth_data_y = train_data.predict_ssh(time)
# predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
# predict_data_x = predict_data_x.cpu().detach().numpy()
# print(predict_data_x.shape, truth_data_y.shape)
# np.save('datasave/3y_3.npy', truth_data_y)
# np.save('datasave/3x_3.npy', predict_data_x)


time1 = 472     # 472       # 预测时间段,注意x和y形状
time2 = 483     # 475
predict_data_x, truth_data_y, piece = train_data.predict_ssh_area(time1, time2)
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()
print(piece, predict_data_x.shape, truth_data_y.shape)
np.save('datasave/y3_19_2.npy', truth_data_y)
np.save('datasave/x3_19_2.npy', predict_data_x)



# def single_pr(x, y, m):
#     x = saved_model(torch.tensor(x, device='cuda'))
#     x = x.cpu().detach().numpy()
#     print(m, x.shape, y.shape)
#     np.save('datasave/y.npy', y)
#     np.save('datasave/x.npy', x)
# single_pr(predict_data_x, truth_data_y, piece)




