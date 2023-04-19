import os
import sys
import argparse
import time
import math
import torch
from utils.util import AverageMeter
from utils.util import adjust_learning_rate, warmup_learning_rate
from utils.util import set_optimizer, save_model
from networks.sigsiam_resnet10 import sigsiam_ResNet10
from networks.sigsiam_convNet import sigsiam_convNet
from networks.sigsiam_DRSN import sigsiam_DRSN_CW
from networks.sigsiam_DRSN_SA import sigsiam_DRSN_SA
from losses import SupConLoss
from datasets.bearing_dataset import BearingSignalDataset



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='600,700,800', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='CWRU_signal_cross_domain', choices=['SEU_gearset', 'SEU_bearing', 'CWRU_signal', 'CWRU_signal_cross_domain', 'spectra_quest_signal'], help='dataset')
    # parser.add_argument('--train_data_txt_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Train_1500_RPM_train.txt')
    parser.add_argument('--train_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_014_021_train.txt')
    # parser.add_argument('--train_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_021_train.txt')

    # method
    parser.add_argument('--method', type=str, default='SigSiam', choices=['SigSiam'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.6, help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--model', type=str, default='convNet', choices=['resnet10', 'convNet','DRSN-CW', 'DRSN-SA'], help='network')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save_net/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_{}'.\
        format(opt.method, opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, time.strftime('%m-%d-%H-%M',time.localtime(time.time())))

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    opt.model_dict = {'resnet10':sigsiam_ResNet10, 'convNet':sigsiam_convNet, 'DRSN-CW':sigsiam_DRSN_CW, 'DRSN-SA':sigsiam_DRSN_SA}

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'spectra_quest_signal':
        labels_list = {
            'N': 0,
            'IR': 1,
            'B': 2,
            'OR': 3,
        }
    elif opt.dataset == 'SEU_bearing':
        labels_list = {
            'N': 0,
            'IR': 1,
            'B': 2,
            'OR': 3,
        }        
    elif opt.dataset == 'SEU_gearset':
        labels_list = {
            'N': 0,
            'Chipped': 1,
            'Miss': 2,
            'Root': 3,
            'Surface': 4,
        }          
    elif opt.dataset == 'CWRU_signal':
        labels_list = {
            'N': 0,
            'IR007': 1,
            'B007': 2,
            'OR007': 3,
            'IR014': 4,
            'B014': 5,
            'OR014': 6,
        }
    elif opt.dataset == 'CWRU_signal_cross_domain':
        labels_list = {
            'N': 0,
            'IR': 1,
            'B': 2,
            'OR': 3
        }            

    else:
        raise ValueError(opt.dataset)


    train_dataset = BearingSignalDataset(opt.train_data_txt_path, labels_list)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                            #    drop_last=True,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    return train_loader


def set_model(opt):
    print("Using " + opt.model +" model!")
    model = opt.model_dict[opt.model]()
    criterion = SupConLoss(temperature=opt.temp)

    model = model.cuda()
    criterion = criterion.cuda()

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (signals, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
    

        if torch.cuda.is_available():
            signals[0] = signals[0].cuda(non_blocking=True)
            signals[1] = signals[1].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)


        # compute loss
        p1, z1, _ = model(signals[0])
        p2, z2, _ = model(signals[1])
        
        features_1 = torch.cat([p1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        features_2 = torch.cat([p2.unsqueeze(1), z1.unsqueeze(1)], dim=1)

        loss = (criterion(features_1, epoch=epoch).mean() + criterion(features_2, epoch=epoch).mean()) * 0.5
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                      epoch,
                      idx + 1,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            if not os.path.isdir(opt.save_folder):
                os.makedirs(opt.save_folder)

            save_file = os.path.join(
                opt.save_folder, 'epoch_{epoch}.pt'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pt')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
