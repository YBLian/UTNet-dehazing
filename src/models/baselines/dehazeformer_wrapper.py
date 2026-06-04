"""
dehazeformer_wrapper.py — 把官方 DehazeFormer 接入本项目的 train_v2/eval_v2 管线
================================================================================
官方 DehazeFormer.forward 已是 [0,1] 输入输出、内部处理任意分辨率、返回单张图。
本 wrapper 仅做两件事:
  1) 修正 sys.path, import 官方 models.dehazeformer 的工厂函数;
  2) 把单输出包成 list [out], 以兼容 train_v2 的多尺度 loss(配 --loss_w 1 即只用主输出)。

接入后 build_model 名称:
  'dehazeformer_t'  -> DehazeFormer-T (depths[4,4,4,2,2], ~0.69M)
  'dehazeformer_s'  -> DehazeFormer-S (depths[8,8,8,4,4], ~1.28M)  ← 与本文 1.342M 最公平
  (b/d/w/m/l 同理可加)

⚠️ 路径: 下面 DEHAZEFORMER_REPO 指向官方仓库根目录, 按你的实际路径改。
"""
import os
import sys
import torch
import torch.nn as nn

# === 官方 DehazeFormer 仓库根目录 (含 models/ datasets/ utils/) ===
DEHAZEFORMER_REPO = r"E:\paperSCI\dehaze\dehaze\dehaze\baselines\DehazeFormer"

if DEHAZEFORMER_REPO not in sys.path:
    sys.path.insert(0, DEHAZEFORMER_REPO)

# 官方工厂函数
from models import (dehazeformer_t, dehazeformer_s, dehazeformer_b,
                    dehazeformer_d, dehazeformer_w, dehazeformer_m, dehazeformer_l)

_FACTORY = {
    'dehazeformer_t': dehazeformer_t,
    'dehazeformer_s': dehazeformer_s,
    'dehazeformer_b': dehazeformer_b,
    'dehazeformer_d': dehazeformer_d,
    'dehazeformer_w': dehazeformer_w,
    'dehazeformer_m': dehazeformer_m,
    'dehazeformer_l': dehazeformer_l,
}


class DehazeFormerWrapper(nn.Module):
    """吃 [0,1] 张量, 返回 list [out([0,1])], 兼容 train_v2 多尺度 loss。"""
    def __init__(self, variant='dehazeformer_s'):
        super().__init__()
        if variant not in _FACTORY:
            raise ValueError(f'未知 DehazeFormer 变体: {variant}; 可选 {list(_FACTORY)}')
        self.net = _FACTORY[variant]()
        self.variant = variant

    def forward(self, x):
        out = self.net(x)                 # 官方: [0,1] -> [0,1], 单张
        return [out]                      # 包成 list 以兼容多尺度 loss(用 --loss_w 1)


def build_dehazeformer(name, **kwargs):
    return DehazeFormerWrapper(variant=name)


if __name__ == '__main__':
    from thop import profile
    for v in ['dehazeformer_t', 'dehazeformer_s']:
        m = DehazeFormerWrapper(v).eval()
        x = torch.randn(1, 3, 64, 64)
        o = m(x)
        p = sum(t.numel() for t in m.parameters())
        f, _ = profile(m.net, inputs=(x,), verbose=False)
        print(f'{v}: params={p/1e6:.3f}M  FLOPs@64={f/1e9:.4f}G  out={tuple(o[0].shape)}')
        for hw in [256, 512]:
            print(f'   full {hw}: {tuple(m(torch.randn(1,3,hw,hw))[0].shape)}')
