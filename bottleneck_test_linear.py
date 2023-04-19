from utils.util import  load_model, set_test_loader
import torch
import argparse
from bottleneck_main_linear import validate
from networks.bottleneck_classifier import feat_bottleneck, feat_classifier
from networks.sigsiam_resnet10 import sigsiam_ResNet10
from networks.sigsiam_convNet import sigsiam_convNet
from networks.sigsiam_DRSN import sigsiam_DRSN_CW
from networks.sigsiam_DRSN_SA import sigsiam_DRSN_SA


def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--print_val_freq', type=int, default=50, help='print val frequency')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # model and dataset
    parser.add_argument('--test_model_path', type=str, default='./save_net/03-17-13-23/50.pt')
    parser.add_argument('--test_bottleneck_path', type=str, default='./save_net/03-17-13-23/bottleneck_50.pt')
    parser.add_argument('--test_classifier_path', type=str, default='./save_net/03-17-13-23/classifier_50.pt')
    # parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/spectra_quest/spectra_quest_Vib_Test_1200_RPM_test.txt')
    # parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_021_test.txt')
    parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/cwru/cwru_DE_1730_RPM_copy_014_test.txt')
    # parser.add_argument('--test_data_txt_path', type=str, default='./data_txt/SEU_gearset/SEU_gearset_20_0_test.txt')

    parser.add_argument('--dataset', type=str, default='CWRU_signal_cross_domain', choices=['SEU_gearset', 'SEU_bearing', 'CWRU_signal', 'CWRU_signal_cross_domain', 'spectra_quest_signal'], help='dataset')



    parser.add_argument('--class_num', type=int, default=4)   
    parser.add_argument('--feature_num', type=int, default=256)
    parser.add_argument('--bottleneck', type=int, default=256)
    # parser.add_argument('--feature_num', type=int, default=512)
    # parser.add_argument('--bottleneck', type=int, default=512)    
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--model', type=str, default='DRSN-SA', choices=['resnet10', 'convNet','DRSN-CW', 'DRSN-SA'], help='network')



    opt = parser.parse_args()

    opt.model_dict = {'resnet10':sigsiam_ResNet10, 'convNet':sigsiam_convNet, 'DRSN-CW':sigsiam_DRSN_CW, 'DRSN-SA':sigsiam_DRSN_SA}

    return opt



def set_model(opt):

    print("Using " + opt.model +" model!")
    model = opt.model_dict[opt.model]()

    bottleneck = feat_bottleneck(feature_dim=opt.feature_num, bottleneck_dim=opt.bottleneck, type=opt.classifier)
    classifier = feat_classifier(class_num=opt.class_num, bottleneck_dim=opt.bottleneck, type=opt.layer)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = classifier.cuda()
    bottleneck = bottleneck.cuda()

    model, criterion = load_model(opt.test_model_path, model, criterion)
    bottleneck = load_model(opt.test_bottleneck_path, bottleneck)
    classifier = load_model(opt.test_classifier_path, classifier)

    return model, bottleneck, classifier, criterion




if __name__ == '__main__':

    opt = parse_option()
    val_loader, labels_dict = set_test_loader(opt)
    model, bottleneck, classifier, criterion = set_model(opt)
    loss, val_acc, macro_f1 = validate(val_loader, model, bottleneck, classifier, criterion, opt, labels_dict)

    print('val_acc accuracy: {:.2f}'.format(val_acc))
    print('val macro_f1: {:.3f}'.format(macro_f1))
    print('val_loss: {:.2f}'.format(loss))