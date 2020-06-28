import torch
import torch.nn as nn


class UnetConcatGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer = nn.modules.normalization.GroupNorm):
        super(UnetConcatGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear")
        self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.reflectionPad = nn.ReflectionPad2d((1, 1, 1, 1))

        self.max_pool = nn.MaxPool2d(7)
        #self.flatten = nn.Flatten()

        self.up_conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv4 = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv3 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv2 = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv1_1 = nn.Conv2d(64, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x_inp = x
        d1 = self.conv1(x_inp)  # 256x256 --> 128x128

        d2 = self.conv2(d1)  # 128x128 --> 64x64
        d2 = self.ac(d2)

        d3 = self.conv3(d2)  # 64x64 --> 32x32
        d3 = self.ac(d3)

        d4 = self.conv4(d3)  # 32x32 --> 16x16
        d4 = self.ac(d4)

        d5 = self.conv5(d4)  # 16x16 --> 8x8
        d5 = self.ac(d5)

        # global_feature
        global_feature = self.max_pool(d5).view(-1, 512)
        print("global feature size: ", global_feature.shape)

        u5 = self.upsample(d5)
        u5 = self.reflectionPad(u5)
        u5 = self.up_conv5(u5)
        u5 = self.ac(u5)

        print(u5.shape)
        print(d5.shape)
        u5 = torch.cat([d4, u5], 1)

        u4 = self.upsample(u5)
        u4 = self.reflectionPad(u4)
        u4 = self.up_conv4(u4)
        u4 = self.ac(u4)
        u4 = torch.cat([d3, u4], 1)

        u3 = self.upsample(u4)
        u3 = self.reflectionPad(u3)
        u3 = self.up_conv3(u3)
        u3 = self.ac(u3)
        u3 = torch.cat([d2, u3], 1)

        u2 = self.upsample(u3)
        u2 = self.reflectionPad(u2)
        u2 = self.up_conv2(u2)
        u2 = self.ac(u2)
        u2 = torch.cat([d1, u2], 1)

        u1 = self.upsample(u2)
        u1 = self.reflectionPad(u1)
        u1 = self.up_conv1(u1)
        u1 = self.ac(u1)

        u1 = self.up_conv1_1(u1)

        return global_feature, u5, u4, u3, u2, u1

