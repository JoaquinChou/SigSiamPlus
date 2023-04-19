import torch.nn as nn
import torch.nn.functional as F
import torch
from thop import profile

class RSBU_CW_BasicBlock(nn.Module):
    # mode: Three kinds of RBUs, 0 indicates that output size is C×W×1,1 indicates that output size is 0.5W×1.
    def __init__(self, in_channel, out_channel, stride=1, mode=0):
        super(RSBU_CW_BasicBlock, self).__init__()
        self.stride = stride
        self.mode = mode
        # change the identity branch output as 0.5W×1
        # conv 1×1
        self.mode_conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=2, bias=False)

        self.br1= nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU())
        self.conv1= nn.Conv2d(in_channel, out_channel, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.br2= nn.Sequential(nn.BatchNorm2d(out_channel), nn.ReLU())
        self.conv2= nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_channel, out_channel)
        self.br3= nn.Sequential(nn.BatchNorm1d(out_channel), nn.ReLU())
        self.fc2 = nn.Linear(out_channel, out_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.mode == 0:
            identity = x
        elif self.mode == 1:
            identity = self.mode_conv(x)
        else:
            print("The mode you input is incorrect!")

            
        x = self.br1(x)
        x = self.conv1(x)
        x = self.br2(x)
        x = self.conv2(x)

        abs_x = torch.abs(x)
        gap_x = self.gap(abs_x)

        gap_x = torch.flatten(gap_x, 1)

        shrink_x = self.fc1(gap_x)
        shrink_x = self.br3(shrink_x)
        shrink_x = self.fc2(shrink_x)
        shrink_x = self.sigmoid(shrink_x)
        thres_x = torch.mul(gap_x, shrink_x)

        thres_x = thres_x.unsqueeze(2).unsqueeze(2)

        # Soft thresholding
        sub_x = abs_x - thres_x
        zeros = sub_x - sub_x
        n_sub_x = torch.max(sub_x, zeros)
        soft_thres_x = torch.mul(torch.sign(x), n_sub_x)
        soft_thres_x = soft_thres_x + identity
        return soft_thres_x


        

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
        
        
# DRSN_CW
class sigsiam_DRSN_CW(nn.Module):
    def __init__(self, in_channel=1, out_channel=64):
        super(sigsiam_DRSN_CW, self).__init__()        
        self.conv1 = nn.Conv2d(in_channel, 4, kernel_size=(3, 1), stride=2, padding=(1, 0), bias=False)
        self.block1 = RSBU_CW_BasicBlock(in_channel=4, out_channel=4, stride=2, mode=1)
        self.block2 = nn.Sequential(RSBU_CW_BasicBlock(in_channel=4, out_channel=4, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=4, out_channel=4, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=4, out_channel=4, stride=1, mode=0))

        self.block3 = RSBU_CW_BasicBlock(in_channel=4, out_channel=16, stride=2, mode=1)
        self.block4 = nn.Sequential(RSBU_CW_BasicBlock(in_channel=16, out_channel=16, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=16, out_channel=16, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=16, out_channel=16, stride=1, mode=0))

        self.block5 = RSBU_CW_BasicBlock(in_channel=16, out_channel=64, stride=2, mode=1)
        self.block6 = nn.Sequential(RSBU_CW_BasicBlock(in_channel=64, out_channel=64, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=64, out_channel=64, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=64, out_channel=64, stride=1, mode=0))

        self.block7 = RSBU_CW_BasicBlock(in_channel=64, out_channel=256, stride=2, mode=1)
        self.block8 = nn.Sequential(RSBU_CW_BasicBlock(in_channel=256, out_channel=256, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=256, out_channel=256, stride=1, mode=0),
        RSBU_CW_BasicBlock(in_channel=256, out_channel=256, stride=1, mode=0))

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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = self.projector(x)
        p = self.predictor(x)
        x = F.normalize(x, dim=1)    
        p = F.normalize(p, dim=1) 


        return p, x.detach(), feature






if __name__ == '__main__':

    net = sigsiam_DRSN_CW()
    input = torch.randn(1, 1, 3600, 1)
    flops, params = profile(net, (input, ))
    print("flops: ", flops, "params: ", params)
    print("flops: %.2fGflops" % (flops / 1e9), "params: %.2fM" % (params / 1e6))