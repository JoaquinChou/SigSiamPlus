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


def get_long_tail_per_cls(total_num, label, cls_num, imb_factor):
    signal_max = total_num / cls_num
    signal_num_per_cls = {}
    for cls_idx in range(cls_num):
        num = signal_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        if num < 1:
            num = 1
        signal_num_per_cls[label[cls_idx]] = (int(num))

    return signal_num_per_cls




if __name__ == '__main__':
    dataset = 'spectra_quest'
    workpiece = 'Vib_Train'
    rpm = '900_RPM'
    labels = ['N', 'IR', 'B', 'OR']
    balance_num = 200
    imb_factor = 0.01
    map_num = get_long_tail_per_cls(balance_num * len(labels), labels, len(labels), imb_factor)
    print("map_num->", map_num)

    npy_data_path = '../data/' + dataset + '/' + workpiece + '/' + rpm + '/'
    save_txt = '../data_txt/' + dataset + '/'

    if not os.path.exists(save_txt):
        os.makedirs(save_txt)
 
    prefix_txt = save_txt + dataset + '_' + workpiece + '_' + rpm + '_imb_' + str(imb_factor)


    for fault in os.listdir(npy_data_path):
        data_list = list()
        for sample in os.listdir(npy_data_path + fault + '/'):
            data_list.append(npy_data_path + fault + '/' + sample)

        random.shuffle(data_list)
        print("Finishing creating the " + npy_data_path + fault + "!")
        print("data_list->", len(data_list))
        train_num = map_num[fault]
        print("train_num->", train_num)
        test_num = 0
        write_txt_by_num(prefix_txt, data_list, train_num, test_num)
