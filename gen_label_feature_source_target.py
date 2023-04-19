import torch
import os
import argparse
from networks.sigsiam_resnet10 import sigsiam_ResNet10
from networks.sigsiam_convNet import sigsiam_convNet
from networks.sigsiam_DRSN import sigsiam_DRSN_CW
from networks.sigsiam_DRSN_SA import sigsiam_DRSN_SA
from utils.util import set_test_loader


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='SigSiam Generating feature txt!')
    parser.add_argument('--model_path', default='./save_target_net/03-18-13-49_seed_2023/100.pt',
    # parser.add_argument('--model_path', default='./save_net/SigSiam_DRSN-SA_spectra_quest_signal_lr_0.1_decay_0.0001_bsz_1024_temp_0.7_03-09-08-22_cosine_warm/last.pt',
                         type=str, help='model_path')
    parser.add_argument('--results_txt', default=None, type=str, help='results_txt')
    parser.add_argument('--val_batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--target_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_021_train.txt')
    parser.add_argument('--source_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_007_014_train.txt')
    parser.add_argument('--dataset', type=str, default='CWRU_signal_cross_domain', choices=['SEU_gearset', 'SEU_bearing', 'CWRU_signal', 'CWRU_signal_cross_domain', 'spectra_quest_signal'], help='dataset')
    parser.add_argument('--model', type=str, default='DRSN-SA', choices=['resnet10', 'convNet','DRSN-CW', 'DRSN-SA'], help='network')

    args = parser.parse_args()        
    args.model_dict = {'resnet10':sigsiam_ResNet10, 'convNet':sigsiam_convNet, 'DRSN-CW':sigsiam_DRSN_CW, 'DRSN-SA':sigsiam_DRSN_SA}

    model_path = args.model_path
    bs = args.val_batch_size
    num_workers = args.num_workers

    # load the model
    print("Using " + args.model +" model!")
    model = args.model_dict[args.model]()
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']
    model = model.cuda()
    model.load_state_dict(state_dict)

    # set test loader
    args.test_data_txt_path = args.target_data_txt_path
    target_val_loader, target_labels_dict = set_test_loader(args)


    args.test_data_txt_path = args.source_data_txt_path
    source_val_loader, source_labels_dict = set_test_loader(args)
    for k in source_labels_dict:
        source_labels_dict[k] += len(source_labels_dict.keys())

    model.eval()
    label_list, feature_list = [], []
    with torch.no_grad():
        for batch_idx, (signals, target, _) in enumerate(target_val_loader):
            if torch.cuda.is_available():
                signals = signals.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            bsz = target.shape[0]
            features = model(signals)[2]
            print(features.shape)
            feature_list += features.tolist()
            label_list += target.tolist()
            if batch_idx == 1005:
                break

    with torch.no_grad():
        for batch_idx, (signals, target, _) in enumerate(source_val_loader):
            if torch.cuda.is_available():
                signals = signals.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            bsz = target.shape[0]
            features = model(signals)[2]
            print(features.shape)
            feature_list += features.tolist()
            label_list += target.tolist()
            if batch_idx == 1005:
                break

    if not os.path.exists('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/'):
        os.makedirs('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/')
    with open('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/' + args.results_txt.split('_')[-1] +'.txt', 'w+') as f:
        for i in range(len(label_list)):
            data = str(label_list[i]) + " " + " ".join('%s' % num
                                                    for num in feature_list[i])
            f.write(data)
            f.write('\n')

