import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CycleFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size, 
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)


class CycleMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L,C = x.shape
        
        x=x.view(B,int(math.sqrt(L)),int(math.sqrt(L)),C)
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        x=x.view(B,L,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CycleBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

class arya_cycleblock(nn.Module):
    def __init__(self, in_channels):
        super(arya_cycleblock, self).__init__()
        self.inchannels = in_channels
        self.cycleblock = CycleBlock(self.inchannels)

    def forward(self, x):
        b,c,h,w=x.size()
        return self.cycleblock(x.view(-1,c).unsqueeze(0)).view(b,c,h,w)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels,:]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class WeightExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(WeightExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout(0.01),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y #avgpool,, linear, gelu, linear, sigmoid, 


class PAWEBlock(nn.Module):
    def __init__(self, in_channels):
        super(PAWEBlock, self).__init__()
        self.position_attention = EfficientAttention(in_channels, in_channels, 8, in_channels)
        self.weight_excitation = WeightExcitationBlock(in_channels)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        position_att = self.position_attention(x)
        weight_extt = self.weight_excitation(x)
        out = position_att + weight_extt * self.weight
        return out

class CAWEBlock(nn.Module):
    def __init__(self, in_channels):
        super(CAWEBlock, self).__init__()
        self.channel_attention = EfficientAttention(in_channels, in_channels, 8, in_channels)
        self.weight_excitation = WeightExcitationBlock(in_channels)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        channel_att = self.channel_attention(x)
        weight_extt = self.weight_excitation(x)
        out = channel_att + weight_extt * self.weight
        return out
    
class DenseConvoBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DenseConvoBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(self._make_layer(self.in_channels,
                                                self.out_channels))
            self.in_channels = self.out_channels

    @staticmethod
    def _make_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Dropout(0.01),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        outputs = [self.layers[0](x)]
        for layer in self.layers[1:]:
            y = layer(sum(outputs))
            outputs.append(y)
        return sum(outputs)


class DenseAtrousBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DenseAtrousBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dilation_rate = i + 2
            self.layers.append(self._make_layer(self.in_channels,
                                                self.out_channels, dilation_rate))
            self.in_channels = self.out_channels

    @staticmethod
    def _make_layer(in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Dropout(0.01),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation, bias=False))

    def forward(self, x):
        outputs = [self.layers[0](x)]
        for layer in self.layers[1:]:
            y = layer(sum(outputs))
            outputs.append(y)
        return sum(outputs)

class PAWEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PAWEConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            PAWEBlock(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.01),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            PAWEBlock(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.01),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class DA_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DA_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dense_deep_block = DenseConvoBlock(in_channels=self.in_channels,
                                               out_channels=self.out_channels,
                                               num_layers=3)
        self.dense_atrous_block = DenseAtrousBlock(in_channels=self.in_channels,
                                                   out_channels=self.out_channels,
                                                   num_layers=3)

    def forward(self, x):
        return sum([self.dense_deep_block(x), self.dense_atrous_block(x)])
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            
            DA_Block(in_channels, out_channels),
            nn.MaxPool2d(2),)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DA_Block(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DA_Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                                #   arya_cycleblock(out_channels))
        

    def forward(self, x):
        return self.conv(x)


class DAWEU_NET(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DAWEU_NET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.cawe1 = nn.Sequential(CAWEBlock(32),arya_cycleblock(32),nn.Sigmoid())
        self.cawe2 = nn.Sequential(CAWEBlock(64),arya_cycleblock(64),nn.Sigmoid())
        self.cawe3 = nn.Sequential(CAWEBlock(128),arya_cycleblock(128),nn.Sigmoid())
        self.cawe4 = nn.Sequential(CAWEBlock(256),arya_cycleblock(256),nn.Sigmoid())
        self.cawe5 = nn.Sequential(CAWEBlock(512),arya_cycleblock(512),nn.Sigmoid())
        self.pawe1 = PAWEBlock(32)
        self.pawe2 = PAWEBlock(64)
        self.pawe3 = PAWEBlock(128)
        self.pawe4 = PAWEBlock(256)
        self.pawe5 = PAWEBlock(512)

        self.inc = (PAWEConv(n_channels, 32))
        self.down0 = (Down(32, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up0 = (Up(1024, 512 // factor, bilinear))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        # x1 = self.pawe1(x1)
        x2 = self.down0(x1)
        # x2 = self.pawe2(x2)
        x3 = self.down1(x2)
        # x3 = self.pawe3(x3)
        x4 = self.down2(x3)
        # x4 = self.pawe4(x4)
        x5 = self.down3(x4)
        # x5 = self.pawe5(x5)
        x6 = self.down4(x5)
        x = self.up0(x6, self.cawe5(x5))
        x = self.up1(x, self.cawe4(x4))
        x = self.up2(x, self.cawe3(x3))
        x = self.up3(x, self.cawe2(x2))
        x = self.up4(x, self.cawe1(x1))
        logits = self.outc(x)
        return logits
    

