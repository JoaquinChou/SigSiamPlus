import torch.nn as nn
import torch
from thop import profile
import torch.nn.functional as F



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 1
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)


        elif self.num_layers == 1:
            x = self.layer4(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=64): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.layer3 = nn.Linear(in_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        # x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        return x 


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


        

        
        
# sigsiam_convNet
class sigsiam_convNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=64):
        super(sigsiam_convNet, self).__init__()        
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

        self.projector = projection_MLP(256)
        self.predictor = prediction_MLP()




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
        x = torch.flatten(p, 1) 
        feature = x
        x = self.projector(x)
        p = self.predictor(x)
        x = F.normalize(x, dim=1)    
        p = F.normalize(p, dim=1) 
        
        return p, x.detach(), feature




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--class_num', type=int, default=4)   
    opt = parser.parse_args()
    net = sigsiam_convNet()
    input = torch.randn(1, 1, 3600, 1)
    flops, params = profile(net, (input, ))
    print("flops: ", flops, "params: ", params)
    print("flops: %.2fGflops" % (flops / 1e9), "params: %.2fM" % (params / 1e6))