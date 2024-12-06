import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
# import fvcore.nn.weight_init as weight_init



class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock,self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch_size, num_channels, num_points)
        batch_size, num_channels, num_points = x.size()

        query = self.query(x).permute(0, 2, 1)  # (batch_size, num_points, num_channels // 8)
        key = self.key(x)  # (batch_size, num_channels // 8, num_points)
        value = self.value(x)  # (batch_size, num_channels, num_points)

        attention = torch.bmm(query, key)  # (batch_size, num_points, num_points)
        attention = F.softmax(attention, dim=-1)  # Normalize along the last axis

        out = torch.bmm(value, attention.permute(0, 2, 1))  # (batch_size, num_channels, num_points)

        return self.gamma * out + x  # Residual connection




class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

        # for layer in [self.g, self.theta, self.phi, self.W]:
        #     weight_init.c2_xavier_fill(layer)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.g, self.theta, self.phi, self.W]:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(self, x):
        batch_size = x[0].size(0)
        out_channels = x[0].size(1)

        g_x = self.g(x[0]).view(batch_size, out_channels // 8, 1)

        theta_x = self.theta(x[1]).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x[1]).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x[0]
        return z