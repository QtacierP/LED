import torch.nn as nn


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding, norm_layer, use_bias, use_dropout=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding (nn.Padding)  -- the instance of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        conv_block += [padding(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [padding(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ISECRETNetwork(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        padding = nn.ReflectionPad2d
        norm_layer = nn.InstanceNorm2d
        use_bias = True
        # Build Head
        head = [padding(3),
                nn.Conv2d(config.input_nc, config.n_filters, kernel_size=7, bias=use_bias),
                norm_layer(config.n_filters),
                nn.ReLU(True)]

        # Build down-sampling
        downs = []
        for i in range(config.n_downs):
            mult = 2 ** i
            downs += [padding(1), nn.Conv2d(config.n_filters * mult,
                                            config.n_filters * mult * 2,
                                            kernel_size=3, stride=2, bias=use_bias),
                      norm_layer(config.n_filters * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** config.n_downs

        neck = []
        # Build res-blocks
        self.in_ch = config.n_filters * mult * 4
        for i in range(config.n_blocks):
            neck += [ResnetBlock(config.n_filters * mult, padding=padding,
                                norm_layer=norm_layer, use_dropout=False,
                                use_bias=use_bias)]

        ups = []
        # Build up-sampling
        for i in range(config.n_downs):
            mult = 2 ** (config.n_downs - i)
            ups += [nn.ConvTranspose2d(config.n_filters * mult,
                                      int(config.n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(config.n_filters * mult / 2)),
                   nn.ReLU(True)]

        importance_ups = []
        # Build unctainty-aware up-sampling
        for i in range(config.n_downs):
            mult = 2 ** (config.n_downs - i)
            importance_ups += [nn.ConvTranspose2d(config.n_filters * mult,
                                      int(config.n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(config.n_filters * mult / 2)),
                   nn.ReLU(True)]

        
        # Build tail
        ups += [padding(3)]
        ups += [nn.Conv2d(config.n_filters, config.output_nc, kernel_size=7, padding=0)]

        ups += [nn.Tanh()]

        # Build importance tail
        importance_ups += [padding(3)]
        importance_ups += [nn.Conv2d(config.n_filters, config.output_nc, kernel_size=7, padding=0)]

        # Make model
        self.head = nn.Sequential(*head)
        self.downs = nn.Sequential(*downs)
        self.neck = nn.Sequential(*neck)
        self.ups = nn.Sequential(*ups)
        self.importance_ups = nn.Sequential(*importance_ups)

    def forward(self, input, need_importance=False, layers=None):
        if layers is None:
            x = self.head(input)
            x = self.downs(x)
            x = self.neck(x)
            output = self.ups(x)
            if need_importance:
                importance = self.importance_ups(x)
                return output, importance
            else:
                return output
        else:
            return self.forward_features(input, layers)

    def forward_features(self, input, layers):
        # We only focus on the encoding part
        feat = input
        feats = []
        layer_id = 0
        for layer in self.head:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        for layer in self.downs:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        for layer in self.neck:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        return feats, feat