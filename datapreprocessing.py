import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.signal as signal
from scipy.fftpack import fft


resample_num = 1024

def proprecessing(path, resample_num):
    data_train = np.empty([resample_num, 120])
    data1 = pd.read_csv(path)
    for j in range(120):
            data2 = data1.iloc[1:, j]

            scaler = preprocessing.StandardScaler()
            data2 = np.array(data2).reshape(-1, 1)
            data2 = scaler.fit_transform(data2.reshape(-1, 1))
            data2 = data2.reshape(-1)

            data2 = data2[round(data2.shape[0] // 20): - round(data2.shape[0] // 20)]
            # 截取中间90%的数据
            d = signal.resample(data2, resample_num)
            # 重采样, 数据长度resample_num = 512

            data_train[ :, j] = d

            # 进行FFT，获得频域数据
            data_train_fft = fft(data_train)

            print([j])
    # print(data_train.shape)
    print('data_train', data_train.shape)
    print('data_train_fft', data_train_fft.shape)

    np.save(r'E:\competition\AnalysisData\data' + '_' + str(resample_num) + '.npy', data_train)
    np.save(r'E:\competition\AnalysisData\data' + '_' + str(resample_num) + 'fft.npy', data_train_fft)


def get_labels(path_with_f_name, resample_num):
    data0 = pd.read_csv(path_with_f_name, header=None, usecols=[1])
    print('data0', data0.shape)
    np.save(r'E:\competition\AnalysisData\data' + '_' + str(resample_num) + '_labels.npy', data0)


proprecessing(path=r'E:\competition\AnalysisData\data_train.csv', resample_num=resample_num)
get_labels(path_with_f_name=r'E:\competition\AnalysisData\label_train.csv',resample_num=resample_num)


