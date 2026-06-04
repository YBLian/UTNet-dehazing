"""
baseline_wrappers.py — 将 FFA-Net / C2PNet 包装成兼容 train.py 的接口
放置位置: E:\paperSCI\dehaze\dehaze\dehaze\code\

接口要求:
  __init__(in_channel=3, channel=32)
  forward(x) -> (ol0, ol1, ol2, ol3)   4个尺度输出

训练命令 (loss_w 必须设为 1,0,0,0，因为 FFA/C2PNet 只有主输出):
  python train.py --model ffa --loss_w 1,0,0,0 --augment gamma --epochs 500 --data_dir ... --save_dir ...
  python train.py --model c2pnet --loss_w 1,0,0,0 --augment gamma --epochs 500 --data_dir ... --save_dir ...
"""

import sys
import os
import importlib.util
import torch
import torch.nn as nn

# ============ 路径配置（根据你的实际路径） ============
FFA_MODEL_PATH = r'E:\paperSCI\dehaze\dehaze\FFA-Net-Master-main\FFA-Net-Master-main\net\models\FFA.py'
C2P_ROOT = r'E:\paperSCI\dehaze\dehaze\C2PNet-main'
# =====================================================


def _load_ffa_class():
    """用 importlib 从文件路径直接加载 FFA 类，避免 models/ 文件夹冲突"""
    spec = importlib.util.spec_from_file_location("FFA_module", FFA_MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FFA


def _load_c2pnet_class():
    """加载 C2PNet 类"""
    if C2P_ROOT not in sys.path:
        sys.path.insert(0, C2P_ROOT)
    from models.C2PNet import C2PNet
    return C2PNet


class FFA_Wrapper(nn.Module):
    """
    FFA-Net 包装器
    - 内部创建 FFA(gps=3, blocks=19)
    - forward 返回 (ol0, ol1, ol2, ol3)，其中 ol1/ol2/ol3 是 dummy 零张量
    - 训练时用 --loss_w 1,0,0,0 即可
    """
    def __init__(self, in_channel=3, channel=32):
        super().__init__()
        FFA_Class = _load_ffa_class()
        self.net = FFA_Class(gps=3, blocks=19)

    def forward(self, x):
        # FFA-Net 只输出一张图 [B, 3, H, W]
        out = self.net(x)

        # 确保 clamp 到合理范围
        ol0 = out.clamp(0, 1)

        # dummy 输出（不参与 loss，因为 loss_w 后三个为 0）
        b = x.size(0)
        device = x.device
        ol1 = torch.zeros(b, 3, 32, 32, device=device)
        ol2 = torch.zeros(b, 3, 16, 16, device=device)
        ol3 = torch.zeros(b, 3, 8, 8, device=device)

        return ol0, ol1, ol2, ol3


class C2PNet_Wrapper(nn.Module):
    """
    C2PNet 包装器
    - 内部创建 C2PNet(gps=3, blocks=19)
    - forward 返回 (ol0, ol1, ol2, ol3)，其中 ol1/ol2/ol3 是 dummy 零张量
    - 训练时用 --loss_w 1,0,0,0 即可
    """
    def __init__(self, in_channel=3, channel=32):
        super().__init__()
        C2PNet_Class = _load_c2pnet_class()
        self.net = C2PNet_Class(gps=3, blocks=19)

    def forward(self, x):
        # C2PNet 只输出一张图 [B, 3, H, W]
        out = self.net(x)

        ol0 = out.clamp(0, 1)

        b = x.size(0)
        device = x.device
        ol1 = torch.zeros(b, 3, 32, 32, device=device)
        ol2 = torch.zeros(b, 3, 16, 16, device=device)
        ol3 = torch.zeros(b, 3, 8, 8, device=device)

        return ol0, ol1, ol2, ol3
