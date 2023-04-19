import math
import numpy as np
import torch
import torch.optim as optim
from datasets.bearing_dataset import NoTransformBearingSignalDataset





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer



def set_optimizer_fine_tuning(backbone, opt):

    optimizer = optim.SGD(backbone.parameters(),
                          lr=opt.backbone_learning_rate,
                          momentum=opt.backbone_momentum,
                          weight_decay=opt.backbone_weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state



def load_model(ckpt, model, criterion=None):
    ckpt = torch.load(ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict


    model = model.cuda()
    model.load_state_dict(state_dict)

    if criterion is None:

        return model
    else:
        criterion = criterion.cuda()

        return model, criterion


def cal_per_class_PR(y_true, y_pred, labels_dict):
    num_2_labels_dict = dict(zip(labels_dict.values(), labels_dict.keys()))
    per_class_recall_dict = dict()
    per_class_precision_dict = dict()

    for k in num_2_labels_dict.keys():
        y_true_temp = np.zeros((y_true.shape))
        y_pred_temp = np.zeros((y_pred.shape))        
        for i in range(len(y_true)):
            if y_true[i] == k:
                y_true_temp[i] = 1

            if y_pred[i] == k:
                y_pred_temp[i] = 1

        inter = (y_pred_temp * y_true_temp).sum()

        per_class_precision_dict[num_2_labels_dict[k]] = inter / y_pred_temp.sum()
        per_class_recall_dict[num_2_labels_dict[k]] = inter / y_true_temp.sum()

    return per_class_precision_dict, per_class_recall_dict





# construct train data loader
def set_train_loader(opt):
    if opt.dataset == 'CWRU_signal':
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

    elif opt.dataset == 'spectra_quest_signal':
        labels_dict = {
                'N': 0,
                'IR': 1,
                'B': 2,
                'OR': 3,
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
        

    train_dataset = NoTransformBearingSignalDataset(opt.train_data_txt_path, labels_dict)
    val_dataset = NoTransformBearingSignalDataset(opt.test_data_txt_path, labels_dict)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers,
                                            pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.val_batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers,
                                             pin_memory=True)


    return train_loader, val_loader, labels_dict



# construct test loader
def set_test_loader(opt):
    if opt.dataset == 'CWRU_signal':
    
        labels_dict = {
            'N': 0,
            'IR007': 1,
            'B007': 2,
            'OR007': 3,
            'IR014': 4,
            'B014': 5,
            'OR014': 6,
        }

    elif opt.dataset == 'spectra_quest_signal':
        labels_dict = {
                'N': 0,
                'IR': 1,
                'B': 2,
                'OR': 3,
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
        

    val_dataset = NoTransformBearingSignalDataset(opt.test_data_txt_path, labels_dict)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=opt.val_batch_size,
                                            shuffle=False,
                                            num_workers=opt.num_workers,
                                            pin_memory=True)


    return val_loader, labels_dict



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75, opt=None):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = opt.weight_decay
        param_group["momentum"] = opt.momentum
        param_group["nesterov"] = True
    return optimizer    



    
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)