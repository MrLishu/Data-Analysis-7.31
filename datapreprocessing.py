import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.signal as signal
from scipy.fftpack import fft

resample_num = 1024
path = r'data\raw\data_train.csv'

data_train = np.empty([120, resample_num])
data = pd.read_csv(path).T

scaler = preprocessing.StandardScaler()
processed_data = scaler.fit_transform(data)
cut_data = processed_data[:, round(data.shape[1] // 20): - round(data.shape[1] // 20)]  # 截取中间90%的数据

for i in range(120):
    sample = processed_data[i]

    # 重采样, 数据长度resample_num = 512
    resample = signal.resample(sample, resample_num)
    data_train[i] = resample

# 进行FFT，获得频域数据
data_train_fft = np.log(np.abs(fft(data_train)))

print('data_train', data_train.shape)
print('data_train_fft', data_train_fft.shape)

np.save(rf'data\processed\{resample_num}.npy', data_train)
np.save(rf'data\processed\{resample_num}fft.npy', data_train_fft)

# 加载label
path_with_f_name = r'data\raw\label_train.csv'

labels = pd.read_csv(path_with_f_name, header=None, usecols=[1]).squeeze()

encoder = preprocessing.LabelEncoder()
code = encoder.fit_transform(labels)

print('labels', labels.shape)
np.save(rf'data\processed\{resample_num}labels.npy', labels)
