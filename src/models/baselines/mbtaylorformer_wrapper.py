"""
mbtaylorformer_wrapper.py — 把官方 MB-TaylorFormer 接入 train_v2/eval_v2 管线
==============================================================================
官方 MB_TaylorFormer.forward: [0,1] 输入 -> self.output(...)+inp_img -> 单张[0,1]输出。
本 wrapper: 修正 sys.path import 模型类, 按官方 -B yml 超参构造, 输出包成 list [out]。

⚠️ 依赖: torchvision 的 DeformConv2d (自带, 无需编译) + einops。
   官方文件头有 `from torchstat import stat`, 若未装会 import 失败 ——
   故本 wrapper 用 importlib 直接加载模块文件, 并在加载前 stub 掉 torchstat。

接入后 build 名称: 'mbtaylorformer_b'
"""
import os, sys, types, importlib.util
import torch
import torch.nn as nn

MBT_REPO = r"E:\paperSCI\dehaze\dehaze\dehaze\baselines\ICCV-2023-MB-TaylorFormer"
MBT_ARCH = os.path.join(MBT_REPO, "basicsr", "models", "archs", "MB_TaylorFormer.py")

# 若未安装 torchstat, 注入一个假的, 避免官方文件头 `from torchstat import stat` 失败
if 'torchstat' not in sys.modules:
    try:
        import torchstat  # noqa
    except Exception:
        fake = types.ModuleType('torchstat')
        fake.stat = lambda *a, **k: None
        sys.modules['torchstat'] = fake

if MBT_REPO not in sys.path:
    sys.path.insert(0, MBT_REPO)

# 直接按文件路径加载模型模块(绕过 basicsr 包级 __init__ 的注册机制)
_spec = importlib.util.spec_from_file_location("mbt_arch", MBT_ARCH)
_mbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mbt)
MB_TaylorFormer = _mbt.MB_TaylorFormer

# 官方 MB-TaylorFormer-B.yml 超参
_CFG_B = dict(
    inp_channels=3, out_channels=3,
    dim=[24, 48, 72, 96],
    num_blocks=[2, 3, 3, 4],
    num_refinement_blocks=2,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='WithBias',
    dual_pixel_task=False,
    num_path=[2, 2, 2, 2],
    qk_norm=0.5,
    offset_clamp=[-3, 3],
)


class MBTaylorFormerWrapper(nn.Module):
    def __init__(self, variant='mbtaylorformer_b'):
        super().__init__()
        cfg = dict(_CFG_B)
        self.net = MB_TaylorFormer(**cfg)
        self.variant = variant

    def forward(self, x):
        return [self.net(x)]   # 单张 -> list, 兼容多尺度 loss(用 --loss_w 1)


def build_mbtaylorformer(name, **kwargs):
    return MBTaylorFormerWrapper(variant=name)


if __name__ == '__main__':
    m = MBTaylorFormerWrapper().eval()
    p = sum(t.numel() for t in m.parameters())
    print(f'MB-TaylorFormer-B: params={p/1e6:.3f}M')
    for hw in [64, 256]:
        try:
            o = m(torch.randn(1, 3, hw, hw))
            print(f'  in {hw} -> out {tuple(o[0].shape)}')
        except Exception as e:
            print(f'  in {hw} FAILED: {e}')
    try:
        from thop import profile
        f, _ = profile(m.net, inputs=(torch.randn(1, 3, 64, 64),), verbose=False)
        print(f'  FLOPs@64 = {f/1e9:.4f}G')
    except Exception as e:
        print(f'  FLOPs skip: {e}')
