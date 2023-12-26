from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models

from networks import DDAMFaceNet

from collections import OrderedDict

from torch.nn import Module

from einops import rearrange
class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class DDAMNet(nn.Module):
    def __init__(self, num_class=7,num_head=4, pretrained=True):
        super(DDAMNet, self).__init__()

        net = DDAMFaceNet.DDAMFaceNet(embedding_size=256, out_h=7, out_w=7)
        
        if pretrained:
            checkpoint = torch.load('/data/2021/code/fer/paper_2021/DAN-main/DAN-main_new/DDAM_main/pretrained/DDAM_msceleb.pt')['state_dict']
        
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[9:] # remove `module.`
                new_state_dict[name] = v     
            model2_dict = net.state_dict()   
            state_dict = {k:v for k,v in new_state_dict.items() if k in model2_dict.keys()} 
            model2_dict.update(state_dict)
            net.load_state_dict(model2_dict) 
    

        self.features = nn.Sequential(*list(net.children())[:-4])
        self.num_head = num_head
        for i in range(int(num_head/2)):
            setattr(self,"cat_head%d" %(2*i), CoordHorAttHead())
            setattr(self,"cat_head%d" %(2*i+1), CoordVerAttHead())         

        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()        
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        
    def forward(self, x):

        x = self.features(x)
        #print(x.shape)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
            
        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
   
        return out, x, heads

        
class CoordHorAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordHorAtt(512,512)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca  
        
class CoordVerAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordVerAtt(512,512)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca  
                     
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
        
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordHorAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordHorAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_h = self.conv2(x_h).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
       

        y = identity * x_h
        y = self.Linear(y)
        y = self.flatten(y)
 
        return y
class CoordVerAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordVerAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_w = self.conv3(x_w).sigmoid()
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w
        y = self.Linear(y)
        y = self.flatten(y)
 
        return y

