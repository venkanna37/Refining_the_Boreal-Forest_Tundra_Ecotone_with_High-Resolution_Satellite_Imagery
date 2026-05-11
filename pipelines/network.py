import torch
import torch.nn as nn


# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, size='small'):
        super().__init__()
        if size == 'small':
            ch = [32, 64, 128, 256, 512]           # smaller 3,834,977 params
        elif size == 'large':
            ch = [64, 128, 256, 512, 1024]        # bigger  15,329,473 params

        self.conv1 = nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(ch[0])
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(ch[1])
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(ch[2])
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(ch[3])
        self.act4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.act5 = nn.ReLU(inplace=True)

        # self.test = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.upconv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Conv2d(ch[4]+ch[3], ch[3], kernel_size=3, padding=0)
        self.bn6 = nn.BatchNorm2d(ch[3])
        self.act6 = nn.ReLU(inplace=True)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv2d(ch[3]+ch[2], ch[2], kernel_size=3, padding=0)
        self.bn7 = nn.BatchNorm2d(ch[2])
        self.act7 = nn.ReLU(inplace=True)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Conv2d(ch[2]+ch[1], ch[1], kernel_size=3, padding=0)
        self.bn8 = nn.BatchNorm2d(ch[1])
        self.act8 = nn.ReLU(inplace=True)

        self.upconv4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = nn.Conv2d(ch[1]+ch[0], ch[0], kernel_size=3, padding=0)
        self.bn9 = nn.BatchNorm2d(ch[0])
        self.act9 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=False)

        # -------- Output --------
        self.out_conv = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def center_crop(self, enc, dec):
        _, _, h, w = dec.shape
        enc = enc[:, :,
        (enc.shape[2] - h) // 2:(enc.shape[2] + h) // 2,
        (enc.shape[3] - w) // 2:(enc.shape[3] + w) // 2]
        return enc

    def forward(self, x):
        e1 = self.act1(self.bn1(self.conv1(x)))

        e2 = self.maxpool(e1)
        e2 = self.act2(self.bn2(self.conv2(e2)))

        e3 = self.maxpool(e2)
        e3 = self.act3(self.bn3(self.conv3(e3)))

        e4 = self.maxpool(e3)
        e4 = self.act4(self.bn4(self.conv4(e4)))

        bottleneck = self.maxpool(e4)
        bottleneck = self.act5(self.bn5(self.conv5(bottleneck)))

        d4 = self.upconv1(bottleneck)
        e4_crop = self.center_crop(e4, d4)
        skip = torch.cat([d4, e4_crop], dim=1)
        d4 = self.act6(self.bn6(self.conv6(skip)))

        d3 = self.upconv2(d4)
        e3_crop = self.center_crop(e3, d3)
        skip = torch.cat([d3, e3_crop], dim=1)
        d3 = self.act7(self.bn7(self.conv7(skip)))

        d2 = self.upconv3(d3)
        e2_crop = self.center_crop(e2, d2)
        skip = torch.cat([d2, e2_crop], dim=1)
        d2 = self.act8(self.bn8(self.conv8(skip)))

        d1 = self.upconv4(d2)
        e1_crop = self.center_crop(e1, d1)
        skip = torch.cat([d1, e1_crop], dim=1)
        d1 = self.act9(self.bn9(self.conv9(skip)))

        out = self.out_conv(d1)

        return out

# model = TinyUNet()
# print(model)
# rand_tensor = torch.randn(4, 1, 256, 256)
# pred = model(rand_tensor)
# print(pred.shape)