import torch
import torch.nn as nn


class Doubleconv(nn.Module):
    def __init__(self,in_channels, out_channels, mid_channels =None):
        super().__init__()
        if mid_channels ==None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Doubleconv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = Doubleconv(in_channels,out_channels,in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = Doubleconv(in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.inc = Doubleconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
    def forward(self,x):     
        r1 = self.inc(x)
        r2 = self.down1(r1)
        r3 = self.down2(r2)
        r4 = self.down3(r3) 
        return r4, r3, r2, r1
    
class Decoder(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(Decoder,self).__init__()
        factor = 2 if bilinear else 1
        
        self.up1 = Up(512, 256// factor, bilinear)
        self.up2 = Up(256, 128// factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = Outconv(64, n_classes)
        
    def forward(self,r4,r3,r2,r1):
        x = self.up1(r4, r3)
        x = self.up2(x, r2)
        x = self.up3(x, r1)
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    def __init__(self, in_channels = 1 ,out_channels = 1):
        super(UNet, self).__init__()
        self.Encoder = Encoder(in_channels)
        self.Decoder = Decoder(out_channels)
        
    def forward(self,x):
        r4, r3, r2, r1 = self.Encoder(x)
        logit = self.Decoder(r4, r3, r2, r1)
        logit = torch.sigmoid(logit)
        return logit
    
def main():
    net = UNet(1,1)
    frames = torch.zeros((1,1,512,512))
    
    out = net(frames)
    print(out.shape)
    
if __name__ == '__main__':
    main()



