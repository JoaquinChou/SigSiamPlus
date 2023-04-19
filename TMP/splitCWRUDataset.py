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
    dataset = 'cwru'
    workpiece = 'DE'
    rpm = '1730_RPM_copy'
    flag_1 = '014'

    npy_data_path = '../data/' + dataset + '/' + workpiece + '/' + rpm + '/' + flag_1 +'/'
    save_txt = '../data_txt/' + dataset + '/'
    # flag_2 = '014'
    # flag_3 = '021'
    # flag = False

    if not os.path.exists(save_txt):
        os.makedirs(save_txt)
 
    prefix_txt = save_txt + dataset + '_' + workpiece + '_' + rpm


    # train:test = 2:1
    train_num = 200
    test_num = 100
    for fault in os.listdir(npy_data_path + '/'):
        # if fault == 'N':
        #     train_num = 100
        #     test_num = 48
        # else:
        #     train_num = 200
        #     test_num = 100

        data_list = list()

        for sample in os.listdir(npy_data_path + '/' + fault + '/'):
            data_list.append(npy_data_path + '/' + fault + '/' + sample)

        random.shuffle(data_list)
        print("Finishing creating the " + npy_data_path + '/' + fault + "!")
        print(len(data_list))


        write_txt_by_num(prefix_txt + '_' + flag_1, data_list, train_num, test_num)



    # for size in os.listdir(npy_data_path):
    #     for fault in os.listdir(npy_data_path + '/' + size):
    #         data_list = list()

    #         for sample in os.listdir(npy_data_path + '/' + size + '/' + fault + '/'):
    #             data_list.append(npy_data_path + '/' + size + '/' + fault + '/' + sample)

    #         random.shuffle(data_list)
    #         print("Finishing creating the " + npy_data_path + '/' + size + '/' + fault + "!")
    #         print(len(data_list))

    #         if fault == 'N' and flag:
    #             write_txt_by_num(prefix_txt + '_' + flag_1, data_list, train_num, test_num)
    #             write_txt_by_num(prefix_txt + '_' + flag_2, data_list, train_num, test_num)
    #             write_txt_by_num(prefix_txt + '_' + flag_3, data_list, train_num, test_num)
    #             flag = False

    #         elif size == flag_1:
    #             write_txt_by_num(prefix_txt + '_' + flag_1, data_list, train_num, test_num)
    #         elif size == flag_2:
    #             write_txt_by_num(prefix_txt + '_' + flag_2, data_list, train_num, test_num)
    #         else:
    #             write_txt_by_num(prefix_txt + '_' + flag_3, data_list, train_num, test_num)