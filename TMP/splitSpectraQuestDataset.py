import os
import random


def write_txt_by_num(txt_name, data_list, train_num, test_num):
    cnt = 0
    while cnt < train_num + test_num:
        if cnt < train_num:
            with open(txt_name + '_train.txt', 'a+') as f:
                f.write(data_list[cnt] + '\n')
        else:
            with open(txt_name + '_test.txt', 'a+') as f:
                f.write(data_list[cnt] + '\n')

        cnt += 1

if __name__ == '__main__':
    dataset = 'spectra_quest'
    workpiece = 'Vib_Test'
    rpm = '1500_RPM'

    npy_data_path = '../data/' + dataset + '/' + workpiece + '/' + rpm + '/'
    save_txt = '../data_txt/' + dataset + '/'

    train_num = 0
    test_num = 200

    if not os.path.exists(save_txt):
        os.makedirs(save_txt)
 
    prefix_txt = save_txt + dataset + '_' + workpiece + '_' + rpm


    for fault in os.listdir(npy_data_path):
        data_list = list()
        for sample in os.listdir(npy_data_path + fault + '/'):
            data_list.append(npy_data_path + fault + '/' + sample)

        random.shuffle(data_list)
        print("Finishing creating the " + npy_data_path + fault + "!")
        print("data_list->", len(data_list))
        write_txt_by_num(prefix_txt, data_list, train_num, test_num)
