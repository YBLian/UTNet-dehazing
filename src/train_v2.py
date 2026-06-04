"""
train_v2.py — 配合 model_arch_v2.py 的训练脚本
================================================
相对原 train.py 的关键改动：
1. --seed：完整复现性设置（编辑 #3：需报告 random seeds / 重复训练 / 统计方差）。
2. 自适应多尺度 loss：把 GT 下采样到「每个输出头各自的真实分辨率」，
   因此对 hier=False（所有输出同分辨率）或任意 patch_size 都成立，
   不再像原 train.py 那样硬编码 32/16/8。
3. --model 支持任意变体名（含核心消融 sa_ca_nohier、stage/窗口/头数消融），
   通过 model_arch_v2.build_model 解析。

典型命令见文件末尾 / RUNBOOK.md。
"""
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SIRRDataset
from build_any import build_any as build_model


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_loss_weights(s):
    parts = [float(x) for x in s.split(',')]
    return parts


def multiscale_loss(outs, target, criterion, loss_w):
    """outs: [finest, ..., coarsest]; 自适应把 target 下采样到每个 out 的分辨率。
    loss_w 长度不足时，缺省权重补 0（即只用主输出）。"""
    total = None
    for i, o in enumerate(outs):
        w = loss_w[i] if i < len(loss_w) else 0.0
        if w == 0.0:
            continue
        h, wd = o.shape[-2], o.shape[-1]
        if (h, wd) != (target.shape[-2], target.shape[-1]):
            t = nn.functional.interpolate(target, size=(h, wd),
                                          mode='bilinear', align_corners=False)
        else:
            t = target
        term = w * criterion(o, t)
        total = term if total is None else total + term
    return total


def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device} | Seed: {args.seed}')

    model = build_model(args.model, in_channel=3, channel=args.channel).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model} | channel={args.channel} | params={n_params:,} '
          f'({n_params/1e6:.3f}M)')

    loss_w = parse_loss_weights(args.loss_w)
    print(f'Loss weights: {loss_w} | Augment: {args.augment}')

    if args.extra_data:
        # 联合训练: 主数据根(repeat=1) + 若干额外根(各自 repeat)
        from multi_dataset import ConcatOversampleDataset
        roots = [args.data_dir] + args.extra_data
        repeats = [1] + args.extra_repeat
        if len(args.extra_repeat) != len(args.extra_data):
            raise ValueError('--extra_repeat 数量需与 --extra_data 一致')
        train_set = ConcatOversampleDataset(
            roots, repeats, input_label='haze', target_label='gt',
            patch_size=args.patch_size, augment=args.augment)
    else:
        train_set = SIRRDataset(root_dir=args.data_dir, input_label='haze',
                                target_label='gt', patch_size=args.patch_size,
                                augment=args.augment)
    loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f'\nTraining {args.epochs} epochs, BS={args.bs}, LR={args.lr} -> {args.save_dir}\n')

    best = float('inf')
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outs = model(inputs)
            loss = multiscale_loss(outs, targets, criterion, loss_w)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / max(1, len(loader))
        print(f'Epoch [{epoch+1}/{args.epochs}] loss={avg:.6f} '
              f'time={(time.time()-t0)/60:.1f}min')

        ckpt = {'model': model.state_dict(), 'epoch': epoch + 1,
                'loss': avg, 'args': vars(args)}
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))
        if avg < best:
            best = avg
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pth'))

    print(f'\nDone. best_loss={best:.6f} saved in {args.save_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser('UTNet v2 training')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--model', default='sa_ca',
                   help="sa_ca|sa|ca|sa_ca_nohier|sa_ca_st2|sa_ca_ws4|sa_ca_hd8 ...")
    p.add_argument('--channel', type=int, default=32)
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--loss_w', default='1,0.2,0.04,0.008',
                   help='finest->coarsest 权重(逗号分隔); 不足补0')
    p.add_argument('--augment', default='gamma',
                   choices=['none', 'flip', 'gamma_only', 'gamma'])
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--extra_data', nargs='*', default=None,
                   help='联合训练: 额外数据根(各含 haze/ gt/), 可多个')
    p.add_argument('--extra_repeat', nargs='*', type=int, default=[],
                   help='与 --extra_data 一一对应的过采样倍数, 如 30')
    train(p.parse_args())
