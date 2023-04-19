import sys
import argparse
import time
import torch
import os
from utils.util import AverageMeter
from utils.util import accuracy, cal_per_class_PR, GradientReversal
from utils.util import set_optimizer_fine_tuning, load_model, save_model, op_copy, lr_scheduler
from networks.bottleneck_classifier import feat_bottleneck, feat_classifier
from networks.sigsiam_resnet10 import sigsiam_ResNet10
from networks.sigsiam_convNet import sigsiam_convNet
from networks.sigsiam_DRSN import sigsiam_DRSN_CW
from networks.sigsiam_DRSN_SA import sigsiam_DRSN_SA
from datasets.bearing_dataset import NoTransformBearingSignalDataset
import numpy as np
from sklearn.metrics import f1_score
import random
# import wandb
from losses import CrossEntropyLabelSmooth
import torch.optim as optim
from losses import mmd_rbf_noaccelerate
import math
from torch import nn
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--print_val_freq', type=int, default=50, help='print val frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--backbone_learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--backbone_weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--backbone_momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--save_folder', type=str, default='./save_net/')
    parser.add_argument('--dataset', type=str, default='CWRU_signal_cross_domain', choices=['SEU_gearset', 'SEU_bearing', 'CWRU_signal', 'CWRU_signal_cross_domain', 'spectra_quest_signal'], help='dataset')
    # parser.add_argument('--source_train_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Train_1500_RPM_train.txt')
    # parser.add_argument('--target_train_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Train_1200_RPM_train.txt')
    # parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Test_1200_RPM_test.txt')
    parser.add_argument('--source_train_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_007_014_train.txt')
    parser.add_argument('--target_train_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_021_train.txt')
    parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_021_test.txt')

    # other setting
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--class_num', type=int, default=4)   
    parser.add_argument('--feature_num', type=int, default=256)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--model', type=str, default='DRSN-SA', choices=['resnet10', 'convNet','DRSN-CW', 'DRSN-SA'], help='network')

    opt = parser.parse_args()
    opt.model_dict = {'resnet10':sigsiam_ResNet10, 'convNet':sigsiam_convNet, 'DRSN-CW':sigsiam_DRSN_CW, 'DRSN-SA':sigsiam_DRSN_SA}


    opt.record_time = time.localtime(time.time())
    opt.save_folder += '{}'.format((time.strftime('%m-%d-%H-%M', opt.record_time)))

    
    return opt



def set_loader(opt):
    # construct data loader
    if opt.dataset == 'spectra_quest_signal':
        labels_dict = {
            'N': 0,
            'IR': 1,
            'B': 2,
            'OR': 3,
        }
    elif opt.dataset == 'CWRU_signal':
        labels_dict = {
            'N': 0,
            'IR007': 1,
            'B007': 2,
            'OR007': 3,
            'IR014': 4,
            'B014': 5,
            'OR014': 6,
        }        
    elif opt.dataset == 'SEU_bearing':
        labels_dict = {
            'N': 0,
            'IR': 1,
            'B': 2,
            'OR': 3,
        }        
    elif opt.dataset == 'SEU_gearset':
        labels_dict = {
            'N': 0,
            'Chipped': 1,
            'Miss': 2,
            'Root': 3,
            'Surface': 4,
        }        
    elif opt.dataset == 'CWRU_signal_cross_domain':
        labels_dict = {
            'N': 0,
            'IR': 1,
            'B': 2,
            'OR': 3
        }        
    else:
        raise ValueError(opt.dataset)

    source_train_dataset = NoTransformBearingSignalDataset(opt.source_train_path, labels_dict)
    target_train_dataset = NoTransformBearingSignalDataset(opt.target_train_path, labels_dict)
    val_dataset = NoTransformBearingSignalDataset(opt.test_data_txt_path, labels_dict)

    train_sampler = None
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset,
        batch_size=opt.batch_size,
        #    drop_last=True,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)
    
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset,
        batch_size=opt.batch_size,
        #    drop_last=True,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.val_batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers,
                                             pin_memory=True)


    return source_train_loader, target_train_loader, val_loader, labels_dict




def set_model(opt):
    print("Using " + opt.model +" model!")
    model = opt.model_dict[opt.model]()

    if opt.smooth > 0:
        criterion = CrossEntropyLabelSmooth(num_classes=opt.class_num, epsilon=opt.smooth)
        
    else:   
        criterion = torch.nn.CrossEntropyLoss()


    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(256, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    model = model.cuda()
    criterion = criterion.cuda()

    bottleneck = feat_bottleneck(feature_dim=opt.feature_num, bottleneck_dim=opt.bottleneck, type=opt.classifier)
    classifier = feat_classifier(class_num=opt.class_num, bottleneck_dim=opt.bottleneck, type=opt.layer)
    classifier = classifier.cuda()
    bottleneck = bottleneck.cuda()
    discriminator = discriminator.cuda()


    return model, bottleneck, classifier, discriminator, criterion


def set_optimizer(opt, bottleneck, classifier, discriminator):
    param_group_b = []
    param_group_c = []

    for k, v in bottleneck.named_parameters():
        param_group_b += [{"params": v, "lr": opt.backbone_learning_rate * 1}]  # 1

    for k, v in discriminator.named_parameters():
        param_group_b += [{"params": v, "lr": opt.learning_rate * 1}]  # 1

    for k, v in classifier.named_parameters():
        param_group_c += [{"params": v, "lr": opt.learning_rate * 1}]  # 1


    optimizer_b = optim.Adam(param_group_b)
    optimizer_b = op_copy(optimizer_b)
    optimizer_c = optim.Adam(param_group_c)
    optimizer_c = op_copy(optimizer_c)


    return optimizer_b, optimizer_c


def train(source_train_loader, target_train_loader, model, bottleneck, classifier, discriminator, criterion, optimizer_backbone, optimizer_b, optimizer_c, epoch, opt, iter_num, max_iter):
    """one epoch training"""
    model.train()
    discriminator.train()
    bottleneck.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    domain_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for (source_singals, source_labels, _), (target_singals, _, _) in zip(source_train_loader, target_train_loader):
        iter_num += 1
        lr_scheduler(optimizer_b, iter_num=iter_num, max_iter=max_iter, opt=opt)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter, opt=opt)        
        data_time.update(time.time() - end)


        signals = torch.cat([source_singals, target_singals])
        signals = signals.float().cuda(non_blocking=True)
        source_labels = source_labels.long().cuda(non_blocking=True)   

        bsz = source_labels.shape[0]
        domain_y = torch.cat([torch.ones(source_singals.shape[0]), torch.zeros(target_singals.shape[0])])
        domain_y = domain_y.cuda(non_blocking=True)

        # compute loss
        features = model(signals)[2]
        output = classifier(bottleneck(features[:source_singals.shape[0]]))
        domain_preds = discriminator(features).squeeze()
        
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)

        loss = criterion(output, source_labels)
 
        loss += 0.1 * domain_loss      

        # update metric
        domain_losses.update(domain_loss.item(), bsz)
        losses.update(loss.item(), bsz)

        acc1, _ = accuracy(output, source_labels, topk=(1, 4))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer_backbone.zero_grad()
        optimizer_b.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()


        optimizer_backbone.step()
        optimizer_b.step()
        optimizer_c.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (iter_num + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'domain_loss {domain_loss.val:.3f} ({domain_loss.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, (iter_num-1) % len(source_train_loader) + 1, len(source_train_loader), batch_time=batch_time,
                   data_time=data_time, domain_loss=domain_losses, loss=losses, top1=top1))
            sys.stdout.flush()

    # save model
    if epoch % opt.save_freq == 0:
        if not os.path.exists(opt.save_folder):
            os.makedirs(opt.save_folder)
        save_file = os.path.join(opt.save_folder, '{epoch}.pt'.format(epoch=epoch))
        save_file_classifier = os.path.join(opt.save_folder, 'classifier_{epoch}.pt'.format(epoch=epoch))
        save_file_bottleneck = os.path.join(opt.save_folder, 'bottleneck_{epoch}.pt'.format(epoch=epoch))

        save_model(model, optimizer_backbone, opt, epoch, save_file)
        save_model(bottleneck, optimizer_b, opt, epoch, save_file_bottleneck)
        save_model(classifier, optimizer_c, opt, epoch, save_file_classifier)


    return losses.avg, top1.avg, iter_num


def validate(val_loader, model, bottleneck, classifier, criterion, opt, labels_dict):                                                                                                     
    """validation"""
    model.eval()
    bottleneck.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    y_true = list()
    y_pred = list()

    with torch.no_grad():
        end = time.time()
        for idx, (signals, labels, _) in enumerate(val_loader):
            signals = signals.float().cuda()
            labels = labels.long().cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(bottleneck(model(signals)[2]))
            loss = criterion(output, labels)

            # update macro_f1
            y_true.append(labels[0].cpu().numpy())
            _, per_y_pred = output.topk(1, 1, True, True)
            y_pred.append(per_y_pred[0][0].cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)
            acc1, _ = accuracy(output, labels, topk=(1, 4))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % opt.print_val_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # cal the P, R, macro f1, acc top1 
    per_class_precision_dict, per_class_recall_dict = cal_per_class_PR(y_true, y_pred, labels_dict)
    print("per_class_precision_dict->", per_class_precision_dict)
    print("per_class_recall_dict->", per_class_recall_dict)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(' * macro_f1 {:.3f}'.format(macro_f1))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg, macro_f1

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
        
    best_acc = 0
    best_macro_f1 = 0

    opt = parse_option()

    # build data loader
    source_train_loader, target_train_loader, val_loader, labels_dict = set_loader(opt)

    # build model and criterion
    model, bottleneck, classifier, discriminator, criterion = set_model(opt)

    # build optimizer
    optimizer_backbone = set_optimizer_fine_tuning(model, opt)
   
    optimizer_b, optimizer_c = set_optimizer(opt, bottleneck, classifier, discriminator)

    # wandb init
    # wandb.init(project="sigsiamPlus", entity="joaquin_chou", name=opt.wandb_name)
    # wandb.config = {
    # "learning_rate": opt.learning_rate,
    # "epochs": opt.epochs,
    # "batch_size": opt.batch_size
    # }
    # wandb.watch(model)


    # training routine
    max_iter = opt.epochs * len(source_train_loader)
    iter_num = 0    
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss, acc, iter_num = train(source_train_loader, target_train_loader, model, bottleneck, classifier, discriminator, criterion,
                          optimizer_backbone, optimizer_b, optimizer_c, epoch, opt, iter_num, max_iter)
        time2 = time.time()
        # wandb.log({"train_loss": loss, "epoch":epoch})
        # wandb.log({"train_acc": acc, "epoch":epoch})    
        for params in optimizer_c.param_groups:
            print("optimizer_c_lr->", params['lr'])       
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        _, val_acc, macro_f1 = validate(val_loader, model, bottleneck, classifier, criterion, opt, labels_dict)
        # wandb.log({"val_acc": val_acc, "epoch":epoch})
        # wandb.log({"val_macro_f1": macro_f1, "epoch":epoch})




        if val_acc > best_acc:
            best_acc = val_acc
            best_macro_f1 = macro_f1

    print('best accuracy: {:.2f}'.format(best_acc))
    print('best macro_f1: {:.3f}'.format(best_macro_f1))
    # wandb.finish()

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pt')
    save_file_bottleneck = os.path.join(opt.save_folder, 'bottleneck_last.pt')
    save_file_classifier = os.path.join(opt.save_folder, 'classifier_last.pt')

    save_model(model, optimizer_backbone, opt, opt.epochs, save_file)  
    save_model(bottleneck, optimizer_b, opt, opt.epochs, save_file_bottleneck)
    save_model(classifier, optimizer_c, opt, opt.epochs, save_file_classifier)

if __name__ == '__main__':
    main()
