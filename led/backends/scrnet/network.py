import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from led.backends.arcnet.network import HFCFilter
# from torchvision.models import resnet18

class UnetCombine2LayerGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, config, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetCombine2LayerGenerator, self).__init__()
        ngf = config.n_filters
        num_downs = config.n_downs
        input_nc = config.input_nc
        output_nc = config.output_nc
        assert num_downs == 8
        unet_block8 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        unet_block7 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block6 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block5 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block4 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer)
        unet_block3 = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer)
        unet_block2 = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer)
        unet_block1 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.down1, self.h_up1, self.up1 = unet_block1.down, unet_block1.up, unet_block1.h_up
        self.down2, self.h_up2, self.up2 = unet_block2.down, unet_block2.up, unet_block2.h_up
        self.down3, self.h_up3, self.up3 = unet_block3.down, unet_block3.up, unet_block3.h_up
        self.down4, self.h_up4, self.up4 = unet_block4.down, unet_block4.up, unet_block4.h_up
        self.down5, self.h_up5, self.up5 = unet_block5.down, unet_block5.up, unet_block5.h_up
        self.down6, self.h_up6, self.up6 = unet_block6.down, unet_block6.up, unet_block6.h_up
        self.down7, self.h_up7, self.up7 = unet_block7.down, unet_block7.up, unet_block7.h_up
        self.down8, self.h_up8, self.up8 = unet_block8.down, unet_block8.up, unet_block8.h_up


    def forward(self, x):
        """Standard forward"""
        # downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # upsample
        h_u8 = self.up8(d8)
        h_u7 = self.up7(torch.cat([h_u8, d7], 1))
        h_u6 = self.up6(torch.cat([h_u7, d6], 1))
        h_u5 = self.up5(torch.cat([h_u6, d5], 1))
        h_u4 = self.up4(torch.cat([h_u5, d4], 1))
        h_u3 = self.up3(torch.cat([h_u4, d3], 1))
        h_u2 = self.up2(torch.cat([h_u3, d2], 1))
        h_u1 = self.up1(torch.cat([h_u2, d1], 1))

        # # upsample
        # TODO: 完善
        u8 = self.h_up8(d8)
        u7 = self.h_up7(torch.cat([h_u8, u8], 1))
        u6 = self.h_up6(torch.cat([h_u7, u7], 1))
        u5 = self.h_up5(torch.cat([h_u6, u6], 1))
        u4 = self.h_up4(torch.cat([h_u5, u5], 1))
        u3 = self.h_up3(torch.cat([h_u4, u4], 1))
        u2 = self.h_up2(torch.cat([h_u3, u3], 1))
        u1 = self.h_up1(torch.cat([h_u2, u2], 1))
        # return u1
        return h_u1, u1


class UnetSkipConnectionBlock(nn.Module):
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
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        h_uprelu = nn.ReLU()
        h_upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # 仅仅修改了这个
            h_upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            h_up = [h_uprelu, h_upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            h_upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            h_up = [h_uprelu, h_upconv, h_upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            h_upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            h_up = [h_uprelu, h_upconv, h_upnorm]
            if use_dropout:
                up = up + [nn.Dropout(0.5)]
                h_up = h_up + [nn.Dropout(0.5)]
        self.up = nn.Sequential(*up)
        self.h_up = nn.Sequential(*h_up)
        self.down = nn.Sequential(*down)

    


class SCRNetwork(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.netG = UnetCombine2LayerGenerator(config)
        self.hfc_filter = HFCFilter(config)

    def forward(self, x, mask):
        x= self.hfc_mul_mask(self.hfc_filter, x, mask)
        fake_TBH, fake_TB = self.netG(x)
        #fake_TBH = (fake_TBH + 1) * mask - 1
        fake_TB = (fake_TB + 1) * mask - 1
        #fake_TB_HFC = self.hfc_mul_mask(self.hfc_filter, fake_TB, mask)
        return fake_TB

    def hfc_mul_mask(self, hfc_filter, image, mask):
        hfc = hfc_filter(image, mask)
        # return hfc
        return (hfc + 1) * mask - 1

