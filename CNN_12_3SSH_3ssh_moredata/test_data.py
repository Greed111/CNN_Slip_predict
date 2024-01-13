import numpy as np
import matplotlib.pyplot as pltt
from load_data import *

data_x_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/otherdata/test_x.npy'
data_y_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/otherdata/test_y.npy'
data_x_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/train_x.npy'
data_y_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_3ssh_moredata/datafile_a/train_y.npy'

train_data = my_dataset(data_x_train, data_y_train, 43)
test_data = my_dataset(data_x_test, data_y_test, 26)

m1, n1 = train_data.predict_ssh(421)
m2, n2 = test_data.predict_ssh(1)
m1 = np.squeeze(m1[0, 2, :, :])
m2 = np.squeeze(m2[0, 2, :, :])
n1 = np.squeeze(n1[0, 2, :, :])
n2 = np.squeeze(n2[0, 2, :, :])
pltt.figure()
pltt.contourf(m1)
pltt.colorbar()
pltt.show()
pltt.figure()
pltt.contourf(m2)
pltt.colorbar()
pltt.show()
pltt.figure()
pltt.contourf(n1)
pltt.colorbar()
pltt.show()
pltt.figure()
pltt.contourf(n2)
pltt.colorbar()
pltt.show()