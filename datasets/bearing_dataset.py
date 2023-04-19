from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class BearingSignalDataset(Dataset):
    def __init__(self, data_txt_path, labels_list):
        super(BearingSignalDataset, self).__init__()
        signal_data = []
        with open(data_txt_path) as f:
            a = f.readlines()
        for line in a:
            signal_data.append(line.strip('\n').replace('..', '.'))

        self.signal_data = signal_data
        self.labels_list = labels_list

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, index):
        np_signal_data = np.load(self.signal_data[index])
        np_signal_data = np_signal_data.reshape(
            (1, np_signal_data.shape[0], 1))
        label_name = self.signal_data[index].split('/')[-2]
        transform = SignalTransform(np_signal_data)
        trans_signals = transform()
        label = self.labels_list[label_name]

        return trans_signals, np.array(label)


class NoTransformBearingSignalDataset(Dataset):
    def __init__(self, data_txt_path, labels_list):
        super(NoTransformBearingSignalDataset, self).__init__()
        signal_data = []
        with open(data_txt_path) as f:
            a = f.readlines()
        for line in a:
            signal_data.append(line.strip('\n').replace('..', '.'))


        self.signal_data = signal_data
        self.labels_list = labels_list

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, index):
        np_signal_data = np.load(self.signal_data[index])
        np_signal_data = np_signal_data.reshape(
            (1, np_signal_data.shape[0], 1))
        label_name = self.signal_data[index].split('/')[-2]
        label = self.labels_list[label_name]

        np_signal_data = (np_signal_data - np.min(np_signal_data)) / (np.max(np_signal_data) - np.min(np_signal_data))
        
        return np_signal_data, np.array(label), index





class SignalTransform:
    """Create two operations of the same signal"""
    def __init__(self, signal_data):
        self.signal_data = signal_data

    def __call__(self):

        
        tran_1 = torch.Tensor(transform_signal_1(self.signal_data))
        tran_2 = torch.Tensor(transform_signal_2(self.signal_data))

        return [(tran_1 - torch.min(tran_1)) /
                (torch.max(tran_1) - torch.min(tran_1)),
                (tran_2 - torch.min(tran_2)) /
                (torch.max(tran_2) - torch.min(tran_2))]



# add the signal transform_1
def transform_signal_1(np_data):

    # Augment_1 TS: Time Swap
    shift = np.random.randint(0, np_data.shape[1] / 2)
    temp = np.copy(np_data[:, 0 : shift + 1])
    np_data[:, 0 : shift + 1] = np_data[:, np_data.shape[1] - shift - 1 : ]
    np_data[:, np_data.shape[1] - shift - 1 : ] = temp

    # Augment_2 AO: Amplitude Offset
    np_data = np_data + np.random.uniform(np.min(np_data), np.max(np_data))

    # Augment_3 FO: Flipping Operation 
    np_data = np.flip(np_data)

    # Augment_4 RCS:  Random Crop Sequence--randomly crop the signal as 3088×1
    rand = np.random.randint(0, 512)
    np_data[:, 0 : rand] = 0
    np_data[:, rand + 3088 :] = 0

    # Augment_5 AWGN: Additive White Gaussian Noise--randomly add 0-1dB Gaussian white noise
    np_data = wgn(np_data, np.random.rand()) + np_data

    return np_data.copy()


# add the signal transform_2
def transform_signal_2(np_data):

    # Augment_1 TS: Time Swap
    shift = np.random.randint(0, np_data.shape[1] / 2)
    temp = np.copy(np_data[:, 0 : shift + 1])
    np_data[:, 0 : shift + 1] = np_data[:, np_data.shape[1] - shift - 1 : ]
    np_data[:, np_data.shape[1] - shift - 1 : ] = temp

    # Augment_2 AO: Amplitude Offset
    np_data = np_data + np.random.uniform(np.min(np_data), np.max(np_data))
  
    # Augment_3 FO: Flipping Operation
    np_data = -np_data

    # Augment_4 RCS:  Random Crop Sequence--randomly crop the signal as 3088×1
    rand = np.random.randint(0, 512)
    np_data[:, 0 : rand] = 0
    np_data[:, rand + 3088 :] = 0

    # Augment_5 AWGN: Additive White Gaussian Noise--randomly add 0-1dB Gaussian white noise
    np_data = wgn(np_data, np.random.rand()) + np_data
        
    return np_data.copy()


# Additive white Gaussian noise
def wgn(x, snr):
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / x.shape[1]
    npower = xpower / snr
    noise_x = np.random.randn(x.shape[1]).reshape(1, x.shape[1], 1)

    return noise_x * np.sqrt(npower)
