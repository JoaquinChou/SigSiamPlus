import torch.nn as nn
import torch.nn.functional as F
import torch
from thop import profile

class convNet_BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(convNet_BasicBlock, self).__init__()

        self.br1= nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU())
        self.conv1= nn.Conv2d(in_channel, out_channel, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.br2= nn.Sequential(nn.BatchNorm2d(out_channel), nn.ReLU())
        self.conv2= nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)


    def forward(self, x):
        x = self.br1(x)
        x = self.conv1(x)
        x = self.br2(x)
        x = self.conv2(x)

        return x


        

        
        
# DRSN
class convNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, opt=None):
        super(convNet, self).__init__()        
        self.conv1 = nn.Conv2d(in_channel, 4, kernel_size=(3, 1), stride=2, padding=(1, 0), bias=False)
        self.block1 = convNet_BasicBlock(in_channel=4, out_channel=4, stride=2)
        self.block2 = nn.Sequential(convNet_BasicBlock(in_channel=4, out_channel=4, stride=1),
        convNet_BasicBlock(in_channel=4, out_channel=4, stride=1),
        convNet_BasicBlock(in_channel=4, out_channel=4, stride=1))

        self.block3 = convNet_BasicBlock(in_channel=4, out_channel=16, stride=2)
        self.block4 = nn.Sequential(convNet_BasicBlock(in_channel=16, out_channel=16, stride=1),
        convNet_BasicBlock(in_channel=16, out_channel=16, stride=1),
        convNet_BasicBlock(in_channel=16, out_channel=16, stride=1))

        self.block5 = convNet_BasicBlock(in_channel=16, out_channel=64, stride=2)
        self.block6 = nn.Sequential(convNet_BasicBlock(in_channel=64, out_channel=64, stride=1),
        convNet_BasicBlock(in_channel=64, out_channel=64, stride=1),
        convNet_BasicBlock(in_channel=64, out_channel=64, stride=1))

        self.block7 = convNet_BasicBlock(in_channel=64, out_channel=256, stride=2)
        self.block8 = nn.Sequential(convNet_BasicBlock(in_channel=256, out_channel=256, stride=1),
        convNet_BasicBlock(in_channel=256, out_channel=256, stride=1),
        convNet_BasicBlock(in_channel=256, out_channel=256, stride=1))

        self.avgpool = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.fully_connect = nn.Linear(256, out_channel)

        self.baselines_drsn_linear = nn.Linear(256, opt.class_num)




    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        p = self.avgpool(x)
        feature = torch.flatten(p, 1) 

        x = self.baselines_drsn_linear(feature)

        return p, x, feature




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--class_num', type=int, default=4)   
    opt = parser.parse_args()
    net = convNet(opt=opt)
    input = torch.randn(1, 1, 3600, 1)
    flops, params = profile(net, (input, ))
    print("flops: ", flops, "params: ", params)
    print("flops: %.2fGflops" % (flops / 1e9), "params: %.2fM" % (params / 1e6))