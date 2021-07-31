import pandas as pd
import numpy as np
import torch
import scipy.signal as signal
from scipy.fftpack import fft
from train import model, device
from datapreprocessing import scaler, encoder, resample_num


data = pd.read_csv(r'data\raw\data_train').T
data = scaler.transform(data)
data = data[:, round(data.shape[1] // 20): - round(data.shape[1] // 20)]  # 截取中间90%的数据

data_final = np.empty([data.shape[0], resample_num])

for i in range(data.shape[0]):
    sample = data[i]

    # 重采样, 数据长度resample_num = 512
    resample = signal.resample(sample, resample_num)
    data_final[i] = resample

data_final_fft = np.log(np.abs(fft(data_final)))
data = torch.from_numpy(data_final_fft)
data = data.float().to(device).unsqueeze(dim=1)
output = model(data).squeeze(-1)
answer = encoder.inverse_transform(output.cpu().numpy())

