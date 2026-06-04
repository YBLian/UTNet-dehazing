"""
build_any.py — 统一模型构建分发器
==================================
按模型名前缀分发:
  'dehazeformer_*' -> DehazeFormer 官方模型 (经 wrapper, 输出包成 list)
  其它             -> model_arch_v2.build_model (本文模型及其消融变体)
train_v2/eval_v2 统一调用 build_any(name, ...)。
"""
from model_arch_v2 import build_model as _build_ours


def build_any(name, in_channel=3, channel=32):
    if name.startswith('dehazeformer_'):
        from dehazeformer_wrapper import build_dehazeformer
        return build_dehazeformer(name)
    if name.startswith('mbtaylorformer'):
        from mbtaylorformer_wrapper import build_mbtaylorformer
        return build_mbtaylorformer(name)
    return _build_ours(name, in_channel=in_channel, channel=channel)
