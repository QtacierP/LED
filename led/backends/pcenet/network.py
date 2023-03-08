import functools
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

class OriUnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(OriUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        conv_d1 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        relu_d1 = nn.LeakyReLU(0.2, inplace=True)
        norm_d1 = norm_layer(inner_nc)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)

        conv_u1 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        relu_u1 = nn.ReLU(inplace=True)
        norm_u1 = norm_layer(outer_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, conv_d1, relu_d1, norm_d1]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, relu_d1, conv_d1, norm_d1]
            up = [uprelu, upconv, upnorm, relu_u1, conv_u1, norm_u1]

        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)


class PCEBackbone(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, need_feature=False):
        super(PCEBackbone, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.in_conv1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                      norm_layer(ngf))
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, kernel_size=1),
            nn.Tanh())

        unet_block4 = OriUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=ngf * 9,
                                                 norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)
        # unet_block4 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block3 = OriUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=ngf * 5, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block2 = OriUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=ngf * 3, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block1 = OriUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=ngf * 1, norm_layer=norm_layer, use_dropout=use_dropout) # add the outermost layer
        self.down1, self.up1 = unet_block1.down, unet_block1.up
        self.down2, self.up2 = unet_block2.down, unet_block2.up
        self.down3, self.up3 = unet_block3.down, unet_block3.up
        self.down4, self.up4 = unet_block4.down, unet_block4.up
        # self.down5, self.up5 = unet_block5.down, unet_block5.up
        self.h_conv1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                     norm_layer(ngf))
        self.h_conv2 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                     norm_layer(ngf))
        self.h_conv3 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                     norm_layer(ngf))

        # self.input_low = input_low
        self.need_feature = need_feature
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.inner_conv = nn.Sequential(nn.Conv2d(ngf * 8 + 3, ngf * 8, kernel_size=3, padding=1, bias=use_bias),
                                 norm_layer(ngf))

    def forward(self, input_list, need_feature=False):
        """Standard forward"""
        x_high, down_h1, down_h2, down_h3, down4 = input_list
        down_h1_feature = self.h_conv1(down_h1)  # 128
        down_h2_feature = self.h_conv2(down_h2)  # 64
        down_h3_feature = self.h_conv3(down_h3)  # 32

        in1 = self.in_conv1(x_high)
        # downsample
        d1 = self.down1(in1)

        d2 = self.down2(torch.cat([d1, down_h1_feature], dim=1))
        d3 = self.down3(torch.cat([d2, down_h2_feature], dim=1))
        d4 = self.down4(torch.cat([d3, down_h3_feature], dim=1))

        d4 = self.leaky_relu(d4)
        d4 = self.inner_conv(torch.cat([d4, down4], 1))

        # upsample
        u4 = self.up4(d4)
        # u4 = self.up4(torch.cat([u5, d4], 1))
        u3 = self.up3(torch.cat([u4, d3], 1))
        u2 = self.up2(torch.cat([u3, d2], 1))
        u1 = self.up1(torch.cat([u2, d1], 1))
        out1 = self.out_conv(torch.cat([u1, in1], 1))
        if need_feature:
            under_activate_features = [in1, d1, d2, d3, d4]
            return out1, under_activate_features
        return 



def get_kernel(kernel_len=16, nsig=10.0):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel


class Gaussian_blur_kernel(nn.Module):
    def __init__(self):
        super(Gaussian_blur_kernel, self).__init__()
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(3, 1, 1, 1)
        # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        x_output = F.conv2d(x, self.weight, groups=x.shape[1])
        return x_output


class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20.0):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        kernel = kernel.repeat(3, 1, 1, 1)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        # self.w = weight.repeat(3, 1, 1, 1)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        # 对三个channel分别做卷积
        # res = []
        # for i in range(x.shape[1]):
        #     res.append(F.conv2d(x[:, i:i+1], self.weight))
        # x_output = torch.cat(res, dim=1)
        x_output = F.conv2d(x, self.weight, groups=x.shape[1])
        return x_output
    
class LP5Layer(nn.Module):
    def __init__(self, filter_width=13, nsig=10, sub_mask=False, ratio=4, is_clamp=True, insert_blur=True, insert_level=False):
        """
        ratio：放大特征图，结合is_clamp使用，去除部分值太小和太大的值，提升对比度
        insert_blur：bool，是否对下采样又上采样的特征做模糊，可以保留更多高频成分
        insert_level：是否根据特征图大小改变insert_blur的高斯模糊核大小，需要设置合理的filter_width和nsig，否则使用默认的5 1，
        filter width越大，保留的高频越多；nsig越大，保留的高频越多
        """
        super(LP5Layer, self).__init__()
        self.gaussian_filter1 = Gaussian_kernel(
            filter_width, nsig=nsig)
        self.gaussian_filter2 = Gaussian_kernel(
            int(filter_width / 2) + int(filter_width / 2) % 2 + 1, nsig=nsig/2)
        self.gaussian_filter3 = Gaussian_kernel(
            int(filter_width / 4) + int(filter_width / 4) % 2 + 1, nsig=nsig/4)
        self.gaussian_filter4 = Gaussian_kernel(
            int(filter_width / 6) + int(filter_width / 6) % 2 + 1, nsig=nsig / 6)
        # self.blur = Gaussian_blur_kernel()
        self.blur = Gaussian_kernel(5, nsig=1)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp
        self.insert_blur = insert_blur
        self.insert_level = insert_level

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)

        # down
        down1 = self.downsample(self.blur(x))
        mask1 = torch.nn.functional.interpolate(mask, scale_factor=1/2, mode='bilinear')
        down2 = self.downsample(self.blur(down1))
        mask2 = torch.nn.functional.interpolate(mask1, scale_factor=1/2, mode='bilinear')
        down3 = self.downsample(self.blur(down2))
        mask3 = torch.nn.functional.interpolate(mask2, scale_factor=1/2, mode='bilinear')
        down4 = self.downsample(self.blur(down3))
        mask4 = torch.nn.functional.interpolate(mask3, scale_factor=1 / 2, mode='bilinear')

        # up
        up1 = torch.nn.functional.interpolate(down1, scale_factor=2, mode='bilinear')
        up2 = torch.nn.functional.interpolate(down2, scale_factor=2, mode='bilinear')
        up3 = torch.nn.functional.interpolate(down3, scale_factor=2, mode='bilinear')
        up4 = torch.nn.functional.interpolate(down4, scale_factor=2, mode='bilinear')

        # low
        if self.insert_blur:
            if self.insert_level:
                x_low = self.gaussian_filter1(up1)
                down1_low = self.gaussian_filter2(up2)
                down2_low = self.gaussian_filter3(up3)
                down3_low = self.gaussian_filter4(up4)
            else:
                x_low = self.blur(up1)
                down1_low = self.blur(up2)
                down2_low = self.blur(up3)
                down3_low = self.blur(up4)
        else:
            x_low = up1
            down1_low = up2
            down2_low = up3
            down3_low = up4

        # high
        h1 = self.sub_low_freq(x, x_low, mask)
        h2 = self.sub_low_freq(down1, down1_low, mask1)
        h3 = self.sub_low_freq(down2, down2_low, mask2)
        h4 = self.sub_low_freq(down3, down3_low, mask3)

        if self.sub_mask:
            down4 = (down4 + 1) * mask4 - 1
        return [h1, h2, h3, h4, down4]

    def sub_low_freq(self, x, low, mask):
        res = self.ratio * (x - low)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1
        return res

class PCENetwork(nn.Module):
    def __init__(self, config):
        super(PCENetwork, self).__init__()
        self.config = config
        self.netG = PCEBackbone(config.input_nc, config.output_nc)
        self.lpls_pyramid = LP5Layer(5, 1, sub_mask=True, insert_level=False)
    

    def forward(self, x, mask):
        x = self.netG(x)
        x = (x + 1) * mask - 1
        return x
        