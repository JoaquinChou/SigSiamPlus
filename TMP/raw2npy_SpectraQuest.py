import os
import scipy.io as scio
import numpy as np
import random
import matplotlib.pyplot as plt

# input: 10000×1
def split_signal(signal, len, frame_move):
    startPos = 0
    res = list()
    print(signal.shape[0])
    while startPos + len < signal.shape[0]:
        segment_signal = signal[startPos:startPos+len, :]
        res.append(segment_signal)
        startPos += frame_move

    random.shuffle(res)
    return res


def save_signal(signal_list, save_path, seq_flag):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for signal in signal_list:
        np.save(save_path + '/' + str(seq_flag) + '.npy', signal)
        print("Finishing " + save_path + '/' + str(seq_flag) + '.npy!')
        seq_flag+=1

    return seq_flag



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






if __name__ == '__main__':
    sample_frames = 3600

    # the sample_rate is 10 KHz
    sample_frequency = 10000
    frame_move = 3600

    # label match number
    labels_list = {
        0: 'N',
        1: 'IR',
        2: 'B',
        3: 'OR',
    }
    data_path = 'D:/Ftp_Server/zgx/data/spectra_quest'
    mode = 'Vib_Test'
    signal_id = '1-转速1500四种故障'
    rpm = '1500_RPM'
    save_path = '../data/spectra_quest/'

    data_path = os.path.join(data_path, signal_id + '.mat')
    save_path = os.path.join(save_path, mode, rpm)

    raw_data = scio.loadmat(data_path)[mode + '_data']
    label = scio.loadmat(data_path)[mode + '_label']


    for i in range(raw_data.shape[0]):
        new_save_path = os.path.join(save_path, labels_list[label[i][0]])
        if not os.path.exists(new_save_path):
            seq_flag = 1
            os.makedirs(new_save_path)
        
        per_raw_data = np.float32(raw_data[i]).reshape(raw_data[i].shape[0], 1)
        res = split_signal(per_raw_data, sample_frames, frame_move)
        seq_flag = save_signal(res, new_save_path, seq_flag)