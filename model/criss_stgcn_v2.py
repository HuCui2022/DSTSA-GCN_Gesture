import copy as cp
import math
import torch
import torch.nn as nn
# from mmcv.runner import load_checkpoint
# from ..utils import cache_checkpoint
from utils import Graph
import pdb
from einops import rearrange
from mmcv.cnn import build_activation_layer, build_norm_layer
from torchvision.ops import SqueezeExcitation
EPS = 1e-4

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)

class dgmstcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat

        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

class criss_gcn_v3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU',
                 fomod='org', # series,parallel,org  并联和串联,原始
                 comod='pair-wise', # dot, pair-wise, self-pairwise
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        self.comod = comod
        self.fomod = fomod
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets        #如果没有指定，则按照1/3 处理。
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        # 添加可学习部分
        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # 1.0 1.4 2.0
        self.b = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        if self.fomod == "org":
            return self.org_forward(x, A)
        if self.fomod == 'series':
            return self.series_forward(x, A)
        if self.fomod == "parallel":
            return self.par_forward(x, A)

    def org_forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        A = self.A      # n,v,v

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)       # self.num_subsets = n
        # * The shape of pre_x is N, K, C, T, V;
        # mid_channels = ratio * outchanels

        # 计算A
        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x
            if self.comod == "pair-wise" or self.comod == 'self-pairwise':
                if not (self.ctr == 'NA' or self.ada == 'NA'):
                    tmp_x = tmp_x.mean(dim=-2, keepdim=True)        # n c t v -> n c 1 v    # todo: cen

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # n k d t v
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # todo: share weights

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            if self.comod == "pair-wise":
                # gat 方法
                diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)          # gat 的方法
            # sat 方法
            elif self.comod == "dot":
                diff = torch.einsum('nkdtv, nkdtu-> nkdvu', x1, x2) * (t**-0.5)
                # diff = diff.reshape(n, self.num_subsets, self.mid_channels, -1, v, v)
                diff = diff[:, :, :, None]
            elif self.comod == "self-pairwise":
                diff = x1.unsqueeze(-1) - x1.unsqueeze(-2)  # gat 的方法

            ada_graph = getattr(self, self.ctr_act)(diff)   # softmax

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            # x1 = x1.mean(dim=-2, keepdim=True)
            # x2 = x2.mean(dim=-2, keepdim=True)
            if self.comod == "dot":
                ada_graph = torch.einsum('nkctv,nkctw->nkvw', x1, x2)*((self.mid_channels*t) ** -0.5)
                ada_graph = ada_graph[:, :, None, None]
            elif self.comod == "pair-wise":
                ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]  # attention 方法
            elif self.comod == "self-pairwise":
                ada_graph = x2.mean(dim=-3, keepdim=True).unsqueeze(-1) - x2.mean(dim=-3, keepdim=True).unsqueeze(-2)

            ada_graph = getattr(self, self.ada_act)(ada_graph)    # softmax or sigmoid or tanh
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A

        # 开始计算 xwa
        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6        # n k c t v v
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        # 并联　90.30
        # x4 = self.tanh(pre_x.mean(-3).mean(1).unsqueeze(-1) - pre_x.mean(-3).mean(1).unsqueeze(-2))    # n t v->  n t v v
        # x2 = torch.einsum('ntwv, nkctv -> nkctw', x4, pre_x)
        # x = x*self.a + x2*self.b

        # 串联
        x = x.reshape(n, -1, t, v)
        # x4 = self.tanh(x.mean(1).unsqueeze(-1) - x.mean(1).unsqueeze(-2))  # n t v -> n t v v
        # x2 = torch.einsum('ntwv, nctv -> nctw', x4, x)
        # x = x*self.a + x2*self.b

        x = self.post(x)
        return self.act(self.bn(x) + res)

    def series_forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        A = self.A      # n,v,v

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)       # self.num_subsets = n
        # * The shape of pre_x is N, K, C, T, V;
        # mid_channels = ratio * outchanels

        # 计算A
        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x
            if self.comod == "pair-wise":
                if not (self.ctr == 'NA' or self.ada == 'NA'):
                    tmp_x = tmp_x.mean(dim=-2, keepdim=True)        # n c t v -> n c 1 v    # todo: cen

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # n k d t v
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # todo: share weights

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            if self.comod == "pair-wise":
                # gat 方法
                diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)          # gat 的方法
            # sat 方法
            elif self.comod == "dot":
                diff = torch.einsum('nkdtv, nkdtu-> nkdvu', x1, x2) * (t**-0.5)
                # diff = diff.reshape(n, self.num_subsets, self.mid_channels, -1, v, v)
                diff = diff[:, :, :, None]
            elif self.comod == "self-pairwise":
                diff = x1.mean(dim=-2, keepdim=True).unsqueeze(-1) - x1.mean(dim=-2, keepdim=True).unsqueeze(-2)  # gat 的方法

            ada_graph = getattr(self, self.ctr_act)(diff)   # softmax

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A1 = ada_graph + A
            pre_x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A1).contiguous()

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            # x1 = x1.mean(dim=-2, keepdim=True)
            # x2 = x2.mean(dim=-2, keepdim=True)
            # 这里共享 todo
            if self.comod == "dot":
                ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)*((self.mid_channels) ** -0.5)
                ada_graph = ada_graph[:, :, None]
            elif self.comod == "pair-wise":
                ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]  # attention 方法
            elif self.comod == "self-pairwise":
                ada_graph = x2.mean(dim=-3, keepdim=True).unsqueeze(-1) - x2.mean(dim=-3, keepdim=True).unsqueeze(-2)

            ada_graph = getattr(self, self.ada_act)(ada_graph)    # softmax or sigmoid or tanh
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A2 = ada_graph + A
            pre_x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A2).contiguous()
        # 开始计算 xwa
        if self.ctr is None and self.ada is None:
            #     assert len(A.shape) == 6        # n k c t v v
            #     # * C, T can be 1
            #     if A.shape[2] == 1 and A.shape[3] == 1:
            #         A = A.squeeze(2).squeeze(2)
            #         x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            #     elif A.shape[2] == 1:
            #         A = A.squeeze(2)
            #         x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            #     elif A.shape[3] == 1:
            #         A = A.squeeze(3)
            #         x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            #     else:
            #         x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
            # else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()
        else:
            x = pre_x
        # 并联　90.30
        # x4 = self.tanh(pre_x.mean(-3).mean(1).unsqueeze(-1) - pre_x.mean(-3).mean(1).unsqueeze(-2))    # n t v->  n t v v
        # x2 = torch.einsum('ntwv, nkctv -> nkctw', x4, pre_x)
        # x = x*self.a + x2*self.b

        # 串联
        x = x.reshape(n, -1, t, v)
        # x4 = self.tanh(x.mean(1).unsqueeze(-1) - x.mean(1).unsqueeze(-2))  # n t v -> n t v v
        # x2 = torch.einsum('ntwv, nctv -> nctw', x4, x)
        # x = x*self.a + x2*self.b

        x = self.post(x)
        return self.act(self.bn(x) + res)

    def par_forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        A = self.A      # n,v,v

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)       # self.num_subsets = n
        # * The shape of pre_x is N, K, C, T, V;
        # mid_channels = ratio * outchanels

        # 计算A
        x1, x2 = None, None
        pre_x1, pre_x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x
            # if self.comod == "pair-wise" or self.comod == 'self-pairwise':
            #     if not (self.ctr == 'NA' or self.ada == 'NA'):
            #         tmp_x = tmp_x.mean(dim=-2, keepdim=True)        # n c t v -> n c 1 v    # todo: cen

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # n k d t v
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # todo: share weights

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            if self.comod == "pair-wise":
                # gat 方法
                diff = x1.mean(dim=-2, keepdim=True).unsqueeze(-1) - x2.mean(dim=-2, keepdim=True).unsqueeze(-2)          # gat 的方法
            # sat 方法
            elif self.comod == "dot":
                diff = torch.einsum('nkdtv, nkdtu-> nkdvu', x1, x2) * (t**-0.5)
                # diff = diff.reshape(n, self.num_subsets, self.mid_channels, -1, v, v)
                diff = diff[:, :, :, None]
            elif self.comod == "self-pairwise":
                diff = x1.mean(dim=-2, keepdim=True).unsqueeze(-1) - x1.mean(dim=-2, keepdim=True).unsqueeze(-2)  # gat 的方法

            ada_graph = getattr(self, self.ctr_act)(diff)   # softmax

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A1 = ada_graph + A
            pre_x1 = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A1).contiguous()

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            # x1 = x1.mean(dim=-2, keepdim=True)
            # x2 = x2.mean(dim=-2, keepdim=True)
            if self.comod == "dot":
                ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)*((self.mid_channels) ** -0.5)
                ada_graph = ada_graph[:, :, None]
            elif self.comod == "pair-wise":
                ada_graph = x1.mean(dim=-3, keepdim=True).unsqueeze(-1) - x2.mean(dim=-3, keepdim=True).unsqueeze(-2)  # attention 方法
            elif self.comod == "self-pairwise":
                ada_graph = x2.mean(dim=-3, keepdim=True).unsqueeze(-1) - x2.mean(dim=-3, keepdim=True).unsqueeze(-2)
            ada_graph = getattr(self, self.ada_act)(ada_graph)    # softmax or sigmoid or tanh
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A2 = ada_graph + A
            pre_x2 = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A2).contiguous()



        # 开始计算 xwa
        if self.ctr is None and self.ada is None:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()
        elif self.ctr is not None and self.ada is not None:
            x = pre_x1*self.a + pre_x2*self.b
        else:
            x = pre_x1 or pre_x2

        # 并联　90.30
        # x4 = self.tanh(pre_x.mean(-3).mean(1).unsqueeze(-1) - pre_x.mean(-3).mean(1).unsqueeze(-2))    # n t v->  n t v v
        # x2 = torch.einsum('ntwv, nkctv -> nkctw', x4, pre_x)
        # x = x*self.a + x2*self.b

        # 串联
        x = x.reshape(n, -1, t, v)
        # x4 = self.tanh(x.mean(1).unsqueeze(-1) - x.mean(1).unsqueeze(-2))  # n t v -> n t v v
        # x2 = torch.einsum('ntwv, nctv -> nctw', x4, x)
        # x = x*self.a + x2*self.b

        x = self.post(x)
        return self.act(self.bn(x) + res)

class criss_ctrgcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets        #如果没有指定，则按照1/3 处理。
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        # selayer
        # self.se = ECANet(mid_channels * num_subsets) #90.72
        self.se = CoordAtt(mid_channels * num_subsets, mid_channels * num_subsets)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.pai = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        # 时间尺度
        self.conv3 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        x0 = x
        res = self.down(x)
        A = self.A      # n,v,v
        # 1 (N), K, 1 (C), 1 (T), V, V
        A0 = A[None, :, None, None]
        pre_x = self.pre(x)
        pre_x = self.se(pre_x)
        pre_x = pre_x.reshape(n, self.num_subsets, self.mid_channels, t, v)       # self.num_subsets = n
        x1, x2 = None, None
        tmp_x = x
        # 添加时间gate
        gate_t = tmp_x.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
        gate_t = self.softmax(gate_t)
        tmp_x = tmp_x * gate_t
        tmp_x = tmp_x.mean(dim=-2, keepdim=True)        # n c t v -> n c 1 v    # todo: cen
        x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # n k d t v
        x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)       # todo: share weights

        if self.ctr is not None:
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)          # gat 的方法  n k d t v v
            ada_graph = getattr(self, self.ctr_act)(diff)   # softmax
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A0

        # 开始计算 xwa
        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6        # n k c t v v
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()
        self.staticA = A
        x3 = self.conv3(x0)
        x3 = rearrange(x3, 'n (k d) t v -> n k d t v', k=self.num_subsets)
        x3 = x3.mean(dim=-3, keepdim=True)        # n k 1 t v
        at = x3.unsqueeze(-1) - x3.unsqueeze(-2)        # n k 1 t v v
        A1 = self.tanh(at)
        A1 = A1.squeeze(2)
        # print(A1.shape)
        self.A1 = A1
        x1 = torch.einsum('nkctv, nktvw->nkctw', pre_x, A1).contiguous()
        x = x + x1* self.pai[0]
        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        return self.act(self.bn(x) + res)

class DGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        # baseline
        self.gcn = criss_ctrgcn(in_channels,
                         out_channels,
                         A,
                         ratio=None,
                         ctr='T',
                         ada=None,
                         subset_wise=False,
                         ada_act='softmax',
                         ctr_act='tanh',
                         norm='BN',
                         act='ReLU')
        self.tcn = dgmstcn(out_channels,
                           out_channels,
                           mid_channels=None,
                           num_joints=25,
                           dropout=0.,
                           ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                           stride=stride
                           )
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)  # 普通的时间卷积下采样。

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        # if A is not parmaters, use A,  or use self.A as parameters
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


# x = torch.rand(2, 16, 100, 25).cuda()  # N C T V
# graph_cfg = dict({"layout": "nturgb+d", "mode": "random", "num_filter": 8, "init_off": 0.04, 'init_std': 0.02})
# graph = Graph(**graph_cfg)
# A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
# net = DGBlock(in_channels=16, out_channels=16, A=A, stride=1, residual=True).cuda()
# print(net(x).shape)   # torch.Size([2, 16, 100, 25])
# pdb.set_trace()


class Model(nn.Module):
    def __init__(self,
                 graph_cfg=dict(),
                 num_classes=60,
                 in_channels=3,
                 base_channels=64,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 dropout=0.0,
                 ):
        super().__init__()

        # 建图
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # 标准化
        self.data_bn_type = data_bn_type
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio

        # 增加维度，时间维度下采样。
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [DGBlock(in_channels, base_channels, A.clone(), 1, residual=False)]  # 第一层layer, inc=3,outc=64

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            # 一共10 层stage. 2 -> 10, 9 层。
            stride = 1 + (i in down_stages)  # if stage ==5, 8, stride=2
            in_channels = base_channels  # 64
            if i in inflate_stages:
                inflate_times += 1  # outchannels = basechannel * chartio ** inflate_times, chartio=2, 通道数量在5 8 分别增加2 4 倍率
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels  # 更新base_channels
            modules.append(DGBlock(in_channels, out_channels, A.clone(), stride))     ##### 注意，不要忽略。
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:  # 如果相等，少一层。
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(base_channels, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_class = num_classes
    # def init_weights(self):
    #     if isinstance(self.pretrained, str):
    #         self.pretrained = cache_checkpoint(self.pretrained)
    #         load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x, y=None):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> n m v c t')
        # x = x.permute(0, 1, 3, 4, 2).contiguous()   # n m v c t
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.reshape(N, M * V * C, T))
        else:
            x = self.data_bn(x.reshape(N * M, V * C, T))
        x = x.reshape(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().reshape(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N*M,) + x.shape[1:])
        x = self.pool(x)
        # print(x.shape); pdb.set_trace()
        x = x.reshape(N,M,-1)
        x = x.mean(1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc_cls(x)
        return x, y


if __name__ == "__main__":
    input = torch.rand(2, 3, 64, 20, 1).cuda()
    # graph_args = dict({"labeling_mode": 'spatial'})
    # graph_cfg = dict({"layout": "nturgb+d", "mode": "random", "num_filter": 8, "init_off": 0.04, 'init_std': 0.02})
    graph_cfg = dict({"layout": "ucla", "mode": "random", "num_filter": 3, "init_off": 0.04, 'init_std': 0.02})

    # x = torch.rand(2, 16, 100, 25).cuda()  # N C T V
    # graph_cfg = dict({"layout": "nturgb+d", "mode": "random", "num_filter": 8, "init_off": 0.04, 'init_std': 0.02})
    # graph = Graph(**graph_cfg)
    # A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
    # net = DGBlock(in_channels=16, out_channels=16, A=A, stride=1, residual=True).cuda()
    # print(net(x).shape)   # torch.Size([2, 16, 100, 25])
    # pdb.set_trace()
    net = Model(graph_cfg=graph_cfg,
                num_classes=10,
                in_channels=3,
                base_channels=64,
                ch_ratio=2,
                num_stages=10,
                inflate_stages=[5, 8],
                down_stages=[5, 8],
                data_bn_type='VC',
                num_person=1,
                pretrained=None,
                dropout=0.0,).cuda()
    print(net(input)[0].shape)
    from ptflops import get_model_complexity_info as get_flops

    flops, params = get_flops(net, (3, 150, 20, 1), as_strings=True, print_per_layer_stat=True)
    # indim, outdim, stride
    print(f"flops :{flops},  params : {params}")
