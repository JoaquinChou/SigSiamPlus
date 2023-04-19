import os
import shutil

def mov_file(source_path:str, dest_path:str, cnt:int):
    for file in os.listdir(source_path):
        rename_file = str(cnt) + '.' + file.split('.')[-1]
        shutil.copy(source_path + '/' + file, dest_path + '/' + rename_file)
        print("Finishing " + source_path + '/' + file + "to" + dest_path + '/' + rename_file + " !")
        cnt += 1

    return cnt



if __name__ == '__main__':
    fault = 'OR'
    size_1 = '014'
    size_2 = '021'
    data_path = '../data/cwru/DE/1730_RPM_copy/'
    dest_path = '../data/cwru/DE/1730_RPM_copy/'+ size_1 +'_' + size_2 +'/'
    cnt = 1

    path_1 = os.path.join(data_path, size_1, fault)
    path_2 = os.path.join(data_path, size_2, fault)
    dest_path = os.path.join(dest_path, fault)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    cnt = mov_file(path_1, dest_path, cnt)
    cnt = mov_file(path_2, dest_path, cnt)