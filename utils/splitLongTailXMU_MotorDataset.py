import os
import random
import shutil


def get_long_tail_per_cls(total_num, label, cls_num, imb_factor):
    signal_max = total_num / cls_num
    signal_num_per_cls = {}
    for cls_idx in range(cls_num):
        num = signal_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        if num < 1:
            num = 1
        signal_num_per_cls[label[cls_idx]] = (int(num))
    return signal_num_per_cls


train_dir = 'D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_balance/train/'
long_tail_train_dir = 'D:/Ftp_Server/zgx/data/XMU_Motor_-1dB_data/class_imbalance_imb_0.1/train/'
Labels = ['N', 'BF', 'BRBF', 'SWF', 'ESF', 'RUF']
balance_num = 512
imb_factor = 0.1
# split the long tail data from the balance train data
dict = get_long_tail_per_cls(balance_num * len(Labels), Labels, len(Labels), imb_factor)
balance_num_list = range(1, balance_num + 1)
print(dict)

for folder_dir in os.listdir(train_dir):
    if not os.path.exists(long_tail_train_dir + folder_dir + '/'):
        os.makedirs(long_tail_train_dir + folder_dir + '/')
    random_sample = random.sample(balance_num_list, dict[folder_dir])
    print("folder_dir, random_sample=", folder_dir, random_sample)
    for i in range(len(random_sample)):
        shutil.copyfile(
            train_dir + folder_dir + '/' + str(random_sample[i]) + '.npy',
            long_tail_train_dir + folder_dir + '/' + str(random_sample[i]) +
            '.npy')
        print("Finish copy " + train_dir + folder_dir + '/' +
              str(random_sample[i]) + '.npy' + " to " + long_tail_train_dir +
              folder_dir + '/' + str(random_sample[i]) + '.npy' + '!')
