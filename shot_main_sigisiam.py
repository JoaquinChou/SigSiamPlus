import sys
import argparse
import time
import random
import torch
import numpy as np
from utils.util import AverageMeter
from losses import obtain_label, shot_loss
from utils.util import accuracy, load_model, lr_scheduler, op_copy
from networks.sigsiam_DRSN_SA import sigsiam_DRSN_SA
from networks.bottleneck_classifier import feat_bottleneck, feat_classifier
import torch.optim as optim
import os
from utils.util import save_model, cal_per_class_PR, set_train_loader
from sklearn.metrics import f1_score
import wandb
import torch.nn as nn



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--print_val_freq', type=int, default=50, help='print val frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    # parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.04, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--learning_rate', type=float, default=1, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    # model dataset
    parser.add_argument('--test_model_path', type=str, default='./save_net/01-29-23-58/100.pt')
    parser.add_argument('--test_bottleneck_path', type=str, default='./save_net/01-29-23-58/bottleneck_100.pt')
    parser.add_argument('--test_classifier_path', type=str, default='./save_net/01-29-23-58/classifier_100.pt')

    parser.add_argument('--save_folder', type=str, default='./save_target_net/')
    # parser.add_argument('--train_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_021_train.txt')
    # parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_021_test.txt')
    # parser.add_argument('--train_data_txt_path', type=str, default='./data_txt/SEU_gearset/SEU_gearset_20_0_train.txt')
    # parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/SEU_gearset/SEU_gearset_20_0_test.txt')
    parser.add_argument('--train_data_txt_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Train_1200_RPM_train.txt')
    parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Test_1200_RPM_test.txt')
    # parser.add_argument('--wandb_name', type=str, default='shot_sigsiam_spectra_quest_Vib_Train_1200_RPM_train')
    parser.add_argument('--dataset', type=str, default='spectra_quest_signal', choices=['SEU_gearset', 'SEU_bearing', 'CWRU_signal', 'CWRU_signal_cross_domain', 'spectra_quest_signal'], help='dataset')


    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--pesudo_value', type=float, default=1.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--reg', type=bool, default=False)
    parser.add_argument('--ent', type=bool, default=False)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--class_num', type=int, default=4)   
    parser.add_argument('--feature_num', type=int, default=256)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])

    opt = parser.parse_args()

    opt.record_time = time.localtime(time.time())
    opt.save_folder += '{}_seed_{}'.format((time.strftime('%m-%d-%H-%M', opt.record_time)), opt.seed)

    return opt


def set_model(opt):

    model = sigsiam_DRSN_SA()
    bottleneck = feat_bottleneck(feature_dim=opt.feature_num, bottleneck_dim=opt.bottleneck, type=opt.classifier)
    classifier = feat_classifier(class_num=opt.class_num, bottleneck_dim=opt.bottleneck, type=opt.layer)
    classifier = classifier.cuda()
    bottleneck = bottleneck.cuda()

    model = load_model(opt.test_model_path, model)
    bottleneck = load_model(opt.test_bottleneck_path, bottleneck)
    classifier = load_model(opt.test_classifier_path, classifier)

    return model, bottleneck, classifier



def set_optimizer(opt, model, bottleneck):

    param_group = []
    for k, v in model.named_parameters():
        # param_group += [{'params': v, 'lr': opt.learning_rate * 0.001}]
        param_group += [{'params': v, 'lr': opt.learning_rate}]
    for k, v in bottleneck.named_parameters():
        param_group += [{'params': v, 'lr': opt.learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    return optimizer






def train(train_loader, model, bottleneck, classifier, optimizer, epoch, opt, iter_num, max_iter):
    """one epoch training"""
    model.train()
    bottleneck.train()
    classifier.eval()
    for k, v in classifier.named_parameters():
        v.requires_grad = False

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    bsz = opt.batch_size
    mem_label = None
    num_sample = len(train_loader.dataset)
    score_bank = torch.randn(num_sample, opt.class_num).cuda()        


    # bsz_mem_label = list()
    if opt.pesudo_value > 0:
        model.eval()
        bottleneck.eval()
        predict_score, mem_label = obtain_label(train_loader, model, bottleneck, classifier)
        mem_label = torch.from_numpy(mem_label).long().cuda()
        predict_score = predict_score.cuda()
        model.train()
        bottleneck.train()

    with torch.no_grad():
            iter_test = iter(train_loader)
            for i in range(len(train_loader)):
                data = iter_test.next()
                inputs = data[0]
                indx = data[-1]
                inputs = inputs.cuda()
                output = model(inputs)[2]
                outputs = classifier(bottleneck(output))
                outputs=nn.Softmax(-1)(outputs)
                score_bank[indx] = outputs.detach().clone()  #.cpu()




    for idx, (signals, labels, tar_idx) in enumerate(train_loader):
        data_time.update(time.time() - end)
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, opt=opt)
        # for params in optimizer.param_groups:
        #     print("lr->", params['lr'])   


        signals = signals.float().cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)   

        # compute loss
        output_features = model(signals)[2]
        output = classifier(bottleneck(output_features))


        loss = shot_loss(output, mem_label, predict_score, score_bank, tar_idx, opt)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(output, labels, topk=(1, 4))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    # save model
    if epoch % opt.save_freq == 0 or epoch == opt.epochs:
        if not os.path.exists(opt.save_folder):
            os.makedirs(opt.save_folder)

        save_file = os.path.join(opt.save_folder, '{epoch}.pt'.format(epoch=epoch))
        save_file_classifier = os.path.join(opt.save_folder, 'classifier_{epoch}.pt'.format(epoch=epoch))
 
        save_model(model, optimizer, opt, epoch, save_file)
        torch.save(classifier.state_dict(), save_file_classifier)


    return losses.avg, top1.avg, iter_num



def validate(val_loader, model, bottleneck, classifier, opt, labels_dict):
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
            output_features = model(signals)[2]
            output = classifier(bottleneck(output_features))

            # val loss——cross entropy
            loss = nn.CrossEntropyLoss()(output, labels)

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

    # cal the P, R, macro f1, acc top1 
    per_class_precision_dict, per_class_recall_dict = cal_per_class_PR(y_true, y_pred, labels_dict)
    print("per_class_precision_dict->", per_class_precision_dict)
    print("per_class_recall_dict->", per_class_recall_dict)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(' * macro_f1 {:.3f}'.format(macro_f1))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg, macro_f1



def main():

    
    best_acc = 0
    best_macro_f1 = 0

    opt = parse_option()

    # random seed
    SEED = opt.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # build data loader
    train_loader, val_loader, labels_dict = set_train_loader(opt)

    # build model and criterion
    model, bottleneck, classifier = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model, bottleneck)

    # lr_scheduler
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        # milestones=[2, 4, 6], gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                     T_max=10)
    
    # wandb init
    # wandb.init(project="sigsiamPlus", entity="joaquin_chou", name=opt.wandb_name)
    # wandb.config = {
    # "learning_rate": opt.learning_rate,
    # "epochs": opt.epochs,
    # "batch_size": opt.batch_size
    # }
    # wandb.watch(model)


    # training routine
    max_iter = opt.epochs * len(train_loader)
    iter_num = 0
    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss, acc, iter_num = train(train_loader, model, bottleneck, classifier,
                          optimizer, epoch, opt, iter_num, max_iter)
        time2 = time.time()
        # wandb.log({"train_loss": loss, "epoch":epoch})
        # wandb.log({"train_acc": acc, "epoch":epoch})

        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
            
        # for params in optimizer.param_groups:
        #     print("lr->", params['lr'])        
        # lr_scheduler.step()

        # shot_lr_scheduler(optimizer, epoch-1, opt.epochs)

        loss, val_acc, macro_f1 = validate(val_loader, model, bottleneck, classifier, opt, labels_dict)
        # wandb.log({"val_acc": val_acc, "epoch":epoch})
        # wandb.log({"val_macro_f1": macro_f1, "epoch":epoch})

        if val_acc > best_acc:
            best_acc = val_acc
            best_macro_f1 = macro_f1


    print('best accuracy: {:.2f}'.format(best_acc))
    print('best macro_f1: {:.3f}'.format(best_macro_f1))
    # wandb.finish()


if __name__ == '__main__':
    main()
