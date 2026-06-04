"""
ffa_arch.py — FFA-Net 模型定义
完全匹配 zhilin007/FFA-Net 原始仓库的层名，以兼容预训练权重。

参考论文: FFA-Net: Feature Fusion Attention Network for Single Image Dehazing (AAAI 2020)
原始仓库: https://github.com/zhilin007/FFA-Net

默认参数: gps=3, blocks=19
"""
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias
    )


class PALayer(nn.Module):
    """Pixel Attention Layer"""
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    """Channel Attention Layer"""
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    """FFA基本块: Conv-ReLU-Conv + CA + PA + Residual"""
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    """一组 Block + 1x conv + 全局残差"""
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class FFA(nn.Module):
    """
    FFA-Net 主网络

    Args:
        gps: Group 数量，默认 3
        blocks: 每个 Group 中 Block 数量，默认 19
    """
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3

        pre_process = [conv(3, self.dim, kernel_size)]
        assert gps == 3, "FFA-Net 默认 gps=3"

        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


def load_ffa_pretrained(ckpt_path, device='cuda'):
    """
    加载 FFA-Net 预训练权重

    FFA-Net原始仓库使用 DataParallel 保存，权重key可能带 'module.' 前缀。
    预训练文件格式: .pk 文件 (pickle)

    Args:
        ckpt_path: 预训练权重路径（如 its_train_ffa_3_19.pk 或 ots_train_ffa_3_19.pk）
        device: 设备
    Returns:
        model: 加载好权重的 FFA 模型
    """
    model = FFA(gps=3, blocks=19).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 处理不同的保存格式
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    # 去掉 'module.' 前缀 (DataParallel)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # 尝试 strict=True，如果失败则 strict=False 并打印警告
    try:
        model.load_state_dict(state, strict=True)
        print(f"[FFA-Net] Loaded weights strictly from {ckpt_path}")
    except RuntimeError as e:
        print(f"[FFA-Net] Warning: strict loading failed, trying non-strict: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")

    model.eval()
    return model


if __name__ == '__main__':
    model = FFA(gps=3, blocks=19)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FFA-Net | Params: {n_params:,}")
    inp = torch.randn(1, 3, 64, 64)
    out = model(inp)
    print(f"Input: {inp.shape} -> Output: {out.shape}")
