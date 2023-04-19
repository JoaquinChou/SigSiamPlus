import os
import scipy.io as scio
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.signal as scsignal

# input: 120000×1
def split_signal(signal, len, frame_move):
    startPos = 0
    res = list()
    print(signal.shape[0])
    while startPos + len < signal.shape[0]:
        segment_signal = signal[startPos:startPos+len, :]
        # signal_len += len
        res.append(segment_signal)
        startPos += frame_move

    random.shuffle(res)
    return res


def save_signal(signal_list, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seq_flag = 1
    for signal in signal_list:
        np.save(save_path + '/' + str(seq_flag) + '.npy', signal)
        print("Finishing " + save_path + '/' + str(seq_flag) + '.npy!')
        seq_flag+=1




def plot_signal(sample_frames, sample_frequency, signal, save_img_name):
    time = np.arange(0, sample_frames * 1.0 / sample_frequency,
                 1.0 / sample_frequency)
    # crop the random sample of sample_frames        
    # rand = np.random.randint(0, signal.shape[0] - sample_frames)
    plt.figure()
    plt.plot(time, signal)
    # plt.plot(time, signal[rand : rand + sample_frames, :])
    plt.xlabel("Time(s)", fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel("Amplitude", fontdict={'family': 'Times New Roman', 'size': 16})

    plt.title("Time Domain image",
                fontdict={
                    'family': 'Times New Roman',
                    'size': 16
                })

    plt.savefig(save_img_name)






# split by rotate speed
if __name__ == '__main__':
    sample_frames = 3600
    # the sample_rate is 12 KHz
    # sample_frequency = 12000
    # the sample_rate is 48 KHz
    sample_frequency = 12000
    frame_move = 390

    data_path = 'D:/Ftp_Server/zgx/data/cwru/12k Drive End Bearing Fault Data/'
    # data_path = 'D:/Ftp_Server/zgx/data/cwru/Normal Baseline Data/'
    # data_path = 'D:/Ftp_Server/zgx/data/cwru/12k Fan End Bearing Fault Data/'
    mode = 'DE'
    signal_id = '237'
    fault_type = 'OR021'
    rpm = '1730_RPM'
    save_path = '../data/cwru/'

    data_path = os.path.join(data_path, signal_id + '.mat')
    save_path = os.path.join(save_path, mode, rpm, fault_type)

    raw_data = scio.loadmat(data_path)['X' + signal_id + '_' + mode + '_time']
    raw_data = np.float32(raw_data)
    # 进行降采样
    if fault_type == 'N':
        raw_data = scsignal.resample(raw_data, raw_data.shape[0] // 4)

    res = split_signal(raw_data, sample_frames, frame_move)

    save_signal(res, save_path)