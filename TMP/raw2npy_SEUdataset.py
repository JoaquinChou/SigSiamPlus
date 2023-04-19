import os
import numpy as np
import random
import pandas as pd

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



def save_signal(signal_list, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seq_flag = 1
    for signal in signal_list:
        np.save(save_path + '/' + str(seq_flag) + '.npy', signal)
        print("Finishing " + save_path + '/' + str(seq_flag) + '.npy!')
        seq_flag+=1






if __name__ == '__main__':
    sample_frames = 3600

    # the sample_rate is 10 KHz
    # sample_frequency = 10000
    frame_move = 3600


    data_path = 'D:/Ftp_Server/zgx/data/SEU_Mechanical-datasets/gearbox/gearset/'
    mode = '30_2'
    signal_id = 'Surface_30_2'
    label = 'Surface'
    save_path = '../data/SEU_gearset/'

    data_path = os.path.join(data_path, signal_id + '.csv')
    save_path = os.path.join(save_path, mode, label)

    raw_data = pd.read_csv(data_path, sep='\t')
    print(raw_data)
    raw_data  = np.array(raw_data.iloc[:,0])
    raw_data = raw_data.reshape(raw_data.shape[0], 1)
    raw_data = np.float32(raw_data)

    res = split_signal(raw_data, sample_frames, frame_move)
    save_signal(res, save_path)