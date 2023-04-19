import torch
import torch.nn as nn
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
        



class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=(1, 0))->None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(3, 1),stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True), # 原地替换 节省内存开销
            # nn.Conv2d(out_channels,out_channels,kernel_size=(3, 1),stride=stride[1],padding=padding,bias=False),
            # nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 采用bn的网络中，卷积层的输出并不加偏置
class sigsiam_ResNet10(nn.Module):
    def __init__(self) -> None:
        super(sigsiam_ResNet10, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=(7, 1),stride=2,padding=(3, 0),bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2, padding=(1, 0))
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = projection_MLP(512)
        self.predictor = prediction_MLP()

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        p = self.avgpool(out)
        x = p.reshape(x.shape[0], -1)
        feature = x
        x = self.projector(x)
        p = self.predictor(x)
        x = F.normalize(x, dim=1)    
        p = F.normalize(p, dim=1) 
        
        return p, x.detach(), feature



if __name__ == '__main__':

    net = sigsiam_ResNet10()
    input = torch.randn(1, 1, 3600, 1)
    flops, params = profile(net, (input, ))
    print("flops: ", flops, "params: ", params)
    print("flops: %.2fGflops" % (flops / 1e9), "params: %.2fM" % (params / 1e6))