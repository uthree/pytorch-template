import torch
import torch.nn as nn

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x): # x: [N, C, H, W]
        m = x.mean(dim=1, keepdim=True)
        s = ((x - m) ** 2).mean(dim=1, keepdim=True)
        x = (x - m) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dim_ffn=None, kernel_size=7):
        super(ConvNeXtBlock, self).__init__()
        if dim_ffn == None:
            dim_ffn = channels * 4
        self.c1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, padding_mode='replicate', groups=channels)
        self.norm = ChannelNorm(channels)
        self.c2 = nn.Conv2d(channels, dim_ffn, 1, 1, 0)
        self.gelu = nn.GELU()
        self.c3 = nn.Conv2d(dim_ffn, channels, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.gelu(x)
        x = self.c3(x)
        return x + res

class ResNetBlock(nn.Module):
    def __init__(
            self,
            channels = 64,
            activation = nn.ReLU,
            ):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.c2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act1 = activation()
        self.act2 = activation()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        r = x
        x = self.c1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.c2(x)
        x = self.norm2(x)
        x = self.act2(x) + r
        return x

class MiniBatchStd2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        mb_std = torch.std(x, dim=[0], keepdim=False).unsqueeze(0).mean(dim=1)
        x[:, 0] += mb_std
        return x

class CNN2d(nn.Module):
    def __init__(
            self,
            input_channels = 3,
            stem_kernel_size = 7,
            stem_stride = 2,
            stem_padding = 0,
            output_features = 1000,
            num_blocks_per_stage = [3, 4, 6, 4],
            channels_per_stage = [64, 128, 256, 512],
            block_type = ResNetBlock,
            activation = nn.ReLU,
            minibatch_std = False,
            pooling = 'max', # max or avg
            spectral_norm = False,
            ):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, channels_per_stage[0], stem_kernel_size, stem_stride, stem_padding)
        self.layers = nn.Sequential()
        if pooling == 'max':
            pool = nn.MaxPool2d
        if pooling == 'avg':
            pool = nn.AvgPool2d
        for i, (l, c) in enumerate(zip(num_blocks_per_stage, channels_per_stage)):
            self.layers.append(nn.Sequential(*[block_type(channels=c, activation=activation) for _ in range(l)]))
            if i != len(num_blocks_per_stage) - 1:
                next_channels = channels_per_stage[i+1]
                self.layers.append(nn.Sequential(pool(kernel_size=2), nn.Conv2d(c, next_channels, 1, 1, 0)))
        if minibatch_std:
            self.layers.append(MiniBatchStd2d())
        self.out_fc = nn.Linear(channels_per_stage[-1], output_features)
        if spectral_norm:
            self.stem = nn.utils.spectral_norm(self.stem)
            self.out_fc = nn.utils.spectral_norm(self.out_fc)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = x.mean(dim=(2, 3))
        x = self.out_fc(x)
        return x
