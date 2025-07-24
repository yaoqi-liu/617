import torch
import torch.nn as nn
import math


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 5), 'kernel size must be 3 or 5'
        padding = 2 if kernel_size == 5 else 1

        # 深度可分离卷积
        self.depthwise_conv = nn.Conv3d(2, 2, kernel_size, padding=padding, groups=2, bias=False)
        self.pointwise_conv = nn.Conv3d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return self.sigmoid(x)


class SwiftAttnGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=1):
        super(SwiftAttnGate, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.InstanceNorm3d(F_int),
            nn.ReLU()
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.InstanceNorm3d(F_int),
            nn.ReLU()
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.relu(self.psi(x1 + g1))
        out = g * psi
        out = out + g

        return out

class CMFA(nn.Module):
    def __init__(self, in_dim, out_dim, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.SAG = SwiftAttnGate(in_dim, in_dim, out_dim)
        else:
            self.SAG = nn.Identity()
        self.SA = SpatialAttention()
        self.conv1x1 = nn.Conv3d(in_dim * 2 if not is_bottom else in_dim, in_dim, kernel_size=1)
        self.residual = nn.Identity() if is_bottom else nn.Conv3d(in_dim, in_dim, kernel_size=1)

    def forward(self, x, skip):
        residual = self.residual(x)
        if not self.is_bottom:
            SAG_skip = self.SAG(x, skip)
            x = torch.cat((SAG_skip, x), dim=1)
        else:
            x = self.SAG(x)
        x = self.SA(x) * x
        x = self.conv1x1(x)
        x = x + residual
        return x


if __name__ == '__main__':
    input_sizes = [
        (2, 32, 80, 192, 160),
        (2, 64, 40, 96, 80),
        (2, 128, 20, 48, 40),
        (2, 256, 10, 24, 20),
        (2, 320, 5, 12, 10)
    ]

    for size in input_sizes:
        x = torch.randn(*size).cuda()
        y = torch.randn(*size).cuda()
        in_dim = size[1]
        model = CMFA(in_dim, in_dim, False).cuda()
        out = model(x, y)
        print(f"Input size: {size}, Output size: {out.shape}")
