
import torch
import torch.nn as nn

# from torchsummary import summary

# from torchviz import make_dot
# #import hiddenlayer as hl

class UnetConcatGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer = nn.modules.normalization.GroupNorm):
        super(UnetConcatGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear")
        self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.reflectionPad = nn.ReflectionPad2d((1, 1, 1, 1))

        self.up_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv2 = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv4 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv5 = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv6 = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv7 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.up_conv7_1 = nn.Conv2d(32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

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

        d6 = self.conv5(d5)  # 16x16 --> 8x8
        d6 = self.ac(d6)

        d7 = self.conv6(d6)  # 16x16 --> 8x8
        d7 = self.ac(d7)

        d8 = self.conv7(d7)  # 16x16 --> 8x8
        d8 = self.ac(d8)

        print("d8 shape", d8.shape)

        u1 = self.upsample(d8)
        u1 = self.reflectionPad(u1)
        u1 = self.up_conv1(u1)
        u1 = self.ac(u1)
        u1 = torch.cat([d7, u1], 1)

        u2 = self.upsample(u1)
        u2 = self.reflectionPad(u2)
        u2 = self.up_conv2(u2)
        u2 = self.ac(u2)
        u2 = torch.cat([d6, u2], 1)

        u3 = self.upsample(u2)
        u3 = self.reflectionPad(u3)
        u3 = self.up_conv3(u3)
        u3 = self.ac(u3)
        u3 = torch.cat([d5, u3], 1)

        u4 = self.upsample(u3)
        u4 = self.reflectionPad(u4)
        u4 = self.up_conv3(u4)
        u4 = self.ac(u4)
        u4 = torch.cat([d4, u4], 1)

        u5 = self.upsample(u4)
        u5 = self.reflectionPad(u5)
        u5 = self.up_conv4(u5)
        u5 = self.ac(u5)
        u5 = torch.cat([d3, u5], 1)

        u6 = self.upsample(u5)
        u6 = self.reflectionPad(u6)
        u6 = self.up_conv5(u6)
        u6 = self.ac(u6)
        u6 = torch.cat([d2, u6], 1)

        u7 = self.upsample(u6)
        u7 = self.reflectionPad(u7)
        u7 = self.up_conv6(u7)
        u7 = self.ac(u7)
        u7 = torch.cat([d1, u7], 1)

        u8 = self.upsample(u7)
        u8 = self.reflectionPad(u8)
        u8 = self.up_conv7(u8)
        u8 = self.ac(u8)

        u9 = self.up_conv7_1(u8)



        # return u9, u8, u7, u6, u5, u4, u3, u2, u1
        return u9


# #unet = UnetConcatGenerator()
# BS = 32
# IMG_CHANNEL = 3
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
# # summary(unet, (IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH))
# #
# unet_gen = UnetConcatGenerator(input_nc=3, output_nc=(4), num_downs=5, )
# imgs = torch.randn((BS, IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH))
# ans = make_dot(unet_gen(imgs), params=dict(unet_gen.named_parameters()))

# # import torchvision
# # model = torchvision.models.vgg16()
# # hl_graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
# # hl_graph.save("xxx", format="png")


# # # print(list(unet_gen.children()))
# #
# # ans = unet_gen.forward(imgs)
# # resize_ans = []
# # for i in range(len(ans)):
# #     u = ans[i]
# #     _, _, ori_img_h, ori_img_w = u.shape
# #     scale_f_h, scale_f_w = IMG_HEIGHT/ori_img_h, IMG_WIDTH/ori_img_w
# #     m = nn.UpsamplingBilinear2d(size = (IMG_HEIGHT, IMG_WIDTH))
# #     resize_u = m(u)
# #     resize_ans.append(resize_u)
# #
# # for u in resize_ans:
# #     print(u.shape)




# # ans = make_dot(unet_gen(imgs), params=dict(unet_gen.named_parameters()))
# # print(ans)


