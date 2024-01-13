import numpy as np
from load_data import *
import torch
from model import *

saved_model = torch.load('model/sealevel_move0.pth')
data_x_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/otherdata/test_x.npy'
data_y_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/otherdata/test_y.npy'
test_data = my_dataset(data_x_test, data_y_test, 26)

time1 = 300          # 预测时间段,注意x和y形状
time2 = 303
predict_data_x, truth_data_y, piece = test_data.predict_ssh_area(time1, time2)
predict_data_x = saved_model(torch.tensor(predict_data_x, device='cuda'))
predict_data_x = predict_data_x.cpu().detach().numpy()
print(piece, predict_data_x.shape, truth_data_y.shape)
np.save('datasave/y3_19.npy', truth_data_y)
np.save('datasave/x3_19.npy', predict_data_x)