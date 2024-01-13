import numpy as np
import xarray as xr
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 主体类方法部分
class my_dataset(Dataset):
      def __init__(self, x_path, y_path, time_scale):  # 这里的time_scale指的是在训练、测试情况下，总数除以每组变量数
            super(my_dataset, self).__init__()
            self.data_x = np.load(x_path)
            self.data_y = np.load(y_path)
            self.time = time_scale
      def __getitem__(self, index):
            target_ssh = self.data_y[index]
            sample = self.data_x[index]
            return sample, target_ssh
      def __len__(self):
            return len(self.data_y)

      def predict_ssh(self, time2):
            x = self.data_x[time2]
            y = self.data_y[time2]
            return x.reshape(1, 72, 31, 156), y.reshape(1, 6, 31, 156)

      def predict_ssh_area(self, time_l, time_r):  # 输入想要预测的时间片段索引
            y = self.data_y[time_l:time_r + 1]
            x = self.data_x[time_l:time_r + 1]
            m = time_r - time_l + 1
            return x.reshape(m, 72, 31, 156), y.reshape(m, 6, 31, 156), m

if __name__ == '__main__':
      data_x_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/train_x.npy'
      data_y_train = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/train_y.npy'
      data_x_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/test_x.npy'
      data_y_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_12_3SSH_6ssh_moredata/datafile_a/test_y.npy'
      testtr = my_dataset(data_x_train, data_y_train, 43)
      testts = my_dataset(data_x_test, data_y_test, 9)
      train_data_size = len(testtr)
      test_data_size = len(testts)
      print("测试数据集的长度为：{}".format(test_data_size))
      x1 ,x2 = testtr.predict_ssh(23)  # 范围在1~43
      print(x1.shape, x2.shape)



# if __name__ == '__main__':
#       data_u10_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/u10_test.npy'
#       data_v10_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/v10_test.npy'
#       data_ssh_test = 'E:/pycharm location/py3.10-pytorch2.0/CNN_U_V_SSH_2_ssh/ssh_test.npy'
#       testdata = my_dataset(data_u10_test, data_v10_test, data_ssh_test, 552)
#       print(testdata)
#       x1,x2 = DataLoader(testdata)
#       print(x1.shape,
#             x2.shape)


