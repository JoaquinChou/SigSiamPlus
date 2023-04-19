import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import os

# Additive white Gaussian noise
def wgn(x, snr):
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / x.shape[1]
    npower = xpower / snr
    noise_x = np.random.randn(x.shape[1]).reshape(1, x.shape[1], 1)

    return noise_x * np.sqrt(npower)


# add the signal transform_1
def transform_signal_1(np_data):

    # Augment_1 TS: Time Swap
    shift = np.random.randint(0, np_data.shape[1] / 2)
    temp = np.copy(np_data[:, 0 : shift + 1])
    np_data[:, 0 : shift + 1] = np_data[:, np_data.shape[1] - shift - 1 : ]
    np_data[:, np_data.shape[1] - shift - 1 : ] = temp

    # # Augment_2 AO: Amplitude Offset
    # np_data = np_data + np.random.uniform(np.min(np_data), np.max(np_data))

    # # Augment_3 FO: Flipping Operation 
    # np_data = np.flip(np_data)

    # # Augment_4 RCS:  Random Crop Sequence--randomly crop the signal as 3088×1
    # rand = np.random.randint(0, 512)
    # np_data[:, 0 : rand] = 0
    # np_data[:, rand + 3088 :] = 0

    # # Augment_5 AWGN: Additive White Gaussian Noise--randomly add 0-1dB Gaussian white noise
    # np_data = wgn(np_data, np.random.rand()) + np_data

    return np_data.copy()

if __name__ == '__main__':
    data_path = '../data/spectra_quest/Vib_Train/900_RPM/IR/156.npy'
    figure_dir = '../augment_results/'
    mode = 'origin'
    np_data = np.load(data_path)
    sample_frequency = 48000

    time = np.arange(0, np_data.shape[0] / sample_frequency,
                 1.0 / sample_frequency)

    # np_data = np_data.reshape(1, np_data.shape[0], 1)
    # np_data = transform_signal_1(np_data)
    # np_data = np_data.reshape(np_data.shape[1], 1)
    # print(np_data.shape)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.plot(time, np_data)
    plt.title("原始信号", fontdict={'family': 'SimSun', 'size': 20, 'fontweight': 'bold'})
    plt.xlabel("时间(s)", fontdict={'family': 'SimSun', 'size': 20, 'fontweight': 'bold'})
    plt.ylabel("幅值", fontdict={'family': 'SimSun', 'size': 20, 'fontweight': 'bold'})    
    # plt.axis('off')
    plt.savefig(figure_dir + mode + '.png')
    plt.close()