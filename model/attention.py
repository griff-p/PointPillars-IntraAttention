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

