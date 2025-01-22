import torch
import numpy as np

class LRlayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super().__init__()
        rlu_slope = 0.1
        self.act = torch.nn.LeakyReLU(rlu_slope)
        self.bn = torch.nn.BatchNorm2d(in_feats)
        self.conv = torch.nn.Conv2d(in_feats, out_feats, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    def forward(self, x):
        return self.act(self.conv(self.bn(x)))
class Outlayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super().__init__()
        #rlu_slope = 0.1
        #self.act = torch.nn.LeakyReLU(rlu_slope)
        self.bn = torch.nn.BatchNorm2d(in_feats)
        self.conv = torch.nn.Conv2d(in_feats, out_feats, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    def forward(self, x):
        return (self.conv((x)))
class Tanhlayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super().__init__()
        self.act = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm2d(in_feats)
        self.conv = torch.nn.Conv2d(in_feats, out_feats, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    def forward(self, x):
        #return self.act(self.bn(self.conv(x)))
        return self.act(self.conv(self.bn(x)))
        #return self.act((self.conv((x))))

class BornNet(torch.nn.Module):
    def __init__(self, nfeat0, nfeat1, nfeat2, nfeat3):
        super().__init__()
        self.l00 = LRlayer(1, nfeat0, 3)
        self.l01 = LRlayer(nfeat0, nfeat0, 3)
        self.l02 = LRlayer(nfeat0, nfeat0, 3)

        self.down = torch.nn.MaxPool2d(2)
        #self.down = torch.nn.AvgPool2d(2)

        self.l10 = LRlayer(nfeat0, nfeat1, 3)
        self.l11 = LRlayer(nfeat1, nfeat1, 3)
        self.l12 = LRlayer(nfeat1, nfeat1, 3)

        self.l20 = LRlayer(nfeat1, nfeat2, 3)
        self.l21 = LRlayer(nfeat2, nfeat2, 3)
        self.l22 = LRlayer(nfeat2, nfeat2, 3)

        self.l30 = LRlayer(nfeat2, nfeat3, 3)
        self.l31 = LRlayer(nfeat3, nfeat3, 3)
        self.l32 = LRlayer(nfeat3, nfeat3, 3)

        self.up = torch.nn.Upsample(scale_factor = 2)
        #self.up = torch.nn.Upsample(scale_factor = 2, mode = 'bilinear')

        self.l23 = LRlayer(nfeat3 + nfeat2, nfeat2, 3)
        self.l24 = LRlayer(nfeat2, nfeat2, 3)
        self.l25 = LRlayer(nfeat2, nfeat2, 3)

        self.l13 =LRlayer(nfeat2 + nfeat1, nfeat1, 3)
        self.l14 = LRlayer(nfeat1, nfeat1, 3)
        self.l15 = LRlayer(nfeat1, nfeat1, 3)

        self.l03 = LRlayer(nfeat1 + nfeat0, nfeat0, 3)
        self.l04 = LRlayer(nfeat0, nfeat0, 3)
        self.l05 = LRlayer(nfeat0, nfeat0, 3)

        self.out = Outlayer(nfeat0, 1, 3)
        #self.out = Tanhlayer(nfeat0, 1,3)
    
    def forward(self, x):
        #x = torch.cat((x, torch.roll(x, 1   -1)), axis = 1)
        #print(x.shape)
        x0 = self.l00(x)
        x0 = self.l01(x0)
        x0 = self.l02(x0)

        x1 = self.down(x0)
        x1 = self.l10(x1)
        x1 = self.l11(x1)
        x1 = self.l12(x1)

        x2 = self.down(x1)
        x2 = self.l20(x2)
        x2 = self.l21(x2)
        x2 = self.l22(x2)

        x3 = self.down(x2)
        x3 = self.l30(x3)
        x3 = self.l31(x3)
        x3 = self.l32(x3)
        x3 = self.up(x3)

        x2 = torch.cat((x2, x3), dim = 1)
        x2 = self.l23(x2)
        x2 = self.l24(x2)
        x2 = self.l25(x2)
        x2 = self.up(x2)
        
        x1 = torch.cat((x1, x2), dim = 1)
        x1 = self.l13(x1)
        x1 = self.l14(x1)
        x1 = self.l15(x1)
        x1 = self.up(x1)

        x0 = torch.cat((x0, x1), dim =1)
        x0 = self.l03(x0)
        x0 = self.l04(x0)
        x0 = self.l05(x0)

        return self.out(x0)
    
class ArtNet(torch.nn.Module):
    def __init__(self, nfeat0, nfeat1, nfeat2, nfeat3):
        super().__init__()
        self.l00 = LRlayer(1, nfeat0, 3)
        self.l01 = LRlayer(nfeat0, nfeat0, 3)
        self.l02 = LRlayer(nfeat0, nfeat0, 3)

        self.down = torch.nn.MaxPool2d(2)

        self.l10 = LRlayer(nfeat0, nfeat1, 3)
        self.l11 = LRlayer(nfeat1, nfeat1, 3)
        self.l12 = LRlayer(nfeat1, nfeat1, 3)

        self.l20 = LRlayer(nfeat1, nfeat2, 3)
        self.l21 = LRlayer(nfeat2, nfeat2, 3)
        self.l22 = LRlayer(nfeat2, nfeat2, 3)

        self.l30 = LRlayer(nfeat2, nfeat3, 3)
        self.l31 = LRlayer(nfeat3, nfeat3, 3)
        self.l32 = LRlayer(nfeat3, nfeat3, 3)

        self.up = torch.nn.Upsample(scale_factor = 2)

        self.l23 = LRlayer(nfeat3 + nfeat2, nfeat2, 3)
        self.l24 = LRlayer(nfeat2, nfeat2, 3)
        self.l25 = LRlayer(nfeat2, nfeat2, 3)

        self.l13 =LRlayer(nfeat2 + nfeat1, nfeat1, 3)
        self.l14 = LRlayer(nfeat1, nfeat1, 3)
        self.l15 = LRlayer(nfeat1, nfeat1, 3)

        self.l03 = LRlayer(nfeat1 + nfeat0, nfeat0, 3)
        self.l04 = LRlayer(nfeat0, nfeat0, 3)
        self.l05 = LRlayer(nfeat0, nfeat0, 3)

        self.out = Tanhlayer(nfeat0, 1, 3)
    
    def forward(self, x):
        x0 = self.l00(x)
        x0 = self.l01(x0)
        x0 = self.l02(x0)

        x1 = self.down(x0)
        x1 = self.l10(x1)
        x1 = self.l11(x1)
        x1 = self.l12(x1)

        x2 = self.down(x1)
        x2 = self.l20(x2)
        x2 = self.l21(x2)
        x2 = self.l22(x2)

        x3 = self.down(x2)
        x3 = self.l30(x3)
        x3 = self.l31(x3)
        x3 = self.l32(x3)
        x3 = self.up(x3)

        x2 = torch.cat((x2, x3), dim = 1)
        x2 = self.l23(x2)
        x2 = self.l24(x2)
        x2 = self.l25(x2)
        x2 = self.up(x2)
        
        x1 = torch.cat((x1, x2), dim = 1)
        x1 = self.l13(x1)
        x1 = self.l14(x1)
        x1 = self.l15(x1)
        x1 = self.up(x1)

        x0 = torch.cat((x0, x1), dim =1)
        x0 = self.l03(x0)
        x0 = self.l04(x0)
        x0 = self.l05(x0)

        return self.out(x0)
    
class TumorNet(torch.nn.Module):
    def __init__(self, nfeat0, nfeat1, nfeat2, nfeat3):
        super().__init__()
        self.l00 = LRlayer(1, nfeat0, 3)
        self.l01 = LRlayer(nfeat0, nfeat0, 3)
        self.l02 = LRlayer(nfeat0, nfeat0, 3)

        self.down = torch.nn.MaxPool2d(2)

        self.l10 = LRlayer(nfeat0, nfeat1, 3)
        self.l11 = LRlayer(nfeat1, nfeat1, 3)
        self.l12 = LRlayer(nfeat1, nfeat1, 3)

        self.l20 = LRlayer(nfeat1, nfeat2, 3)
        self.l21 = LRlayer(nfeat2, nfeat2, 3)
        self.l22 = LRlayer(nfeat2, nfeat2, 3)

        self.l30 = LRlayer(nfeat2, nfeat3, 3)
        self.l31 = LRlayer(nfeat3, nfeat3, 3)
        self.l32 = LRlayer(nfeat3, nfeat3, 3)

        self.up = torch.nn.Upsample(scale_factor = 2)

        self.l23 = LRlayer(nfeat3 + nfeat2, nfeat2, 3)
        self.l24 = LRlayer(nfeat2, nfeat2, 3)
        self.l25 = LRlayer(nfeat2, nfeat2, 3)

        self.l13 =LRlayer(nfeat2 + nfeat1, nfeat1, 3)
        self.l14 = LRlayer(nfeat1, nfeat1, 3)
        self.l15 = LRlayer(nfeat1, nfeat1, 3)

        self.l03 = LRlayer(nfeat1 + nfeat0, nfeat0, 3)
        self.l04 = LRlayer(nfeat0, nfeat0, 3)
        self.l05 = LRlayer(nfeat0, nfeat0, 3)

        self.out = Tanhlayer(nfeat0, 1, 3)
    
    def forward(self, x):
        x0 = self.l00(x)
        x0 = self.l01(x0)
        x0 = self.l02(x0)

        x1 = self.down(x0)
        x1 = self.l10(x1)
        x1 = self.l11(x1)
        x1 = self.l12(x1)

        x2 = self.down(x1)
        x2 = self.l20(x2)
        x2 = self.l21(x2)
        x2 = self.l22(x2)

        x3 = self.down(x2)
        x3 = self.l30(x3)
        x3 = self.l31(x3)
        x3 = self.l32(x3)
        x3 = self.up(x3)

        x2 = torch.cat((x2, x3), dim = 1)
        x2 = self.l23(x2)
        x2 = self.l24(x2)
        x2 = self.l25(x2)
        x2 = self.up(x2)
        
        x1 = torch.cat((x1, x2), dim = 1)
        x1 = self.l13(x1)
        x1 = self.l14(x1)
        x1 = self.l15(x1)
        x1 = self.up(x1)

        x0 = torch.cat((x0, x1), dim =1)
        x0 = self.l03(x0)
        x0 = self.l04(x0)
        x0 = self.l05(x0)

        return (self.out(x0) + 1)/2



























