"""
c2pnet_arch.py — C2PNet 模型定义
匹配 YuZheng9/C2PNet 原始仓库的层名，以兼容预训练权重。

参考论文: Curricular Contrastive Regularization for Physics-aware
          Single Image Dehazing (CVPR 2023)
原始仓库: https://github.com/YuZheng9/C2PNet

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
    """Pixel Attention Layer (同FFA-Net)"""
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
    """Channel Attention Layer (同FFA-Net)"""
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


class PDU(nn.Module):
    """
    Physics-aware Dual-branch Unit (C2PNet核心创新)
    基于大气散射模型: I(x) = J(x)*t(x) + A*(1-t(x))
    双分支分别估计 A (大气光) 和 t (透射率)
    """
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        # A分支 (大气光估计)
        self.a_branch = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        # T分支 (透射率估计)
        self.t_branch = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channel // 4, channel, 1, bias=True),
        )

    def forward(self, x):
        d = self.down(x)
        a = self.a_branch(d)
        t = self.t_branch(d)
        out = a * t + (1 - t) * d
        return self.fuse(out)


class Block(nn.Module):
    """C2PNet基本块: Conv-ReLU-PDU + CA + PA + Residual"""
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        self.pdu = PDU(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res = self.pdu(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class C2PNet(nn.Module):
    """
    C2PNet 主网络
    整体架构与FFA-Net一致，但Block中加入了PDU。
    训练时可配合 Curricular Contrastive Regularization (C2R) loss，
    推理时仅需此网络即可。

    Args:
        gps: Group 数量，默认 3
        blocks: 每个 Group 中 Block 数量，默认 19
    """
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super(C2PNet, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3

        pre_process = [conv(3, self.dim, kernel_size)]
        assert gps == 3

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


def load_c2pnet_pretrained(ckpt_path, device='cuda'):
    """
    加载 C2PNet 预训练权重

    C2PNet 原始仓库保存格式: {'model': state_dict, ...}
    预训练文件: ITS.pkl / OTS.pkl

    Args:
        ckpt_path: 预训练权重路径
        device: 设备
    Returns:
        model: 加载好权重的 C2PNet 模型
    """
    model = C2PNet(gps=3, blocks=19).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state = ckpt['model']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt

    state = {k.replace("module.", ""): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
        print(f"[C2PNet] Loaded weights strictly from {ckpt_path}")
    except RuntimeError as e:
        print(f"[C2PNet] Warning: strict loading failed: {e}")
        print("[C2PNet] 如果层名不匹配，请直接使用原始仓库的模型文件。")
        print("[C2PNet] 将原始仓库中的 models/C2PNet.py 复制到本项目下即可。")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    return model


if __name__ == '__main__':
    model = C2PNet(gps=3, blocks=19)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"C2PNet | Params: {n_params:,}")
    inp = torch.randn(1, 3, 64, 64)
    out = model(inp)
    print(f"Input: {inp.shape} -> Output: {out.shape}")
