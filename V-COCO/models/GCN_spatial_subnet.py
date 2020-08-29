import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, Adj, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        A = np.zeros((num_subset, Adj.shape[0], Adj.shape[1])).astype('float')
        for i in range(num_subset):
            A[i, :, :] = Adj
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
            )
        else:
            self.down = lambda x: x

        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y += self.down(x)
        return self.relu(y)

class GCN_spatial_subnet(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels, A):
        super(GCN_spatial_subnet, self).__init__()

        self.l1 = unit_gcn(in_channels, in_channels, A)
        self.l2 = unit_gcn(in_channels, in_channels, A)
        self.l3 = unit_gcn(in_channels, in_channels, A)
        self.l4 = unit_gcn(in_channels, in_channels, A)
        self.l5 = unit_gcn(in_channels, nhidden, A)
        self.l6 = unit_gcn(nhidden, nhidden, A)
        self.l7 = unit_gcn(nhidden, nhidden, A)
        self.l8 = unit_gcn(nhidden, out_channels, A)
        self.l9 = unit_gcn(out_channels, out_channels, A)
        self.l10 = unit_gcn(out_channels, out_channels, A)

    def forward(self, x):
        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        return x
