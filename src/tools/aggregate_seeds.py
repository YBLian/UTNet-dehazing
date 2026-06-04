"""
aggregate_seeds.py — 多种子结果聚合 (编辑 #3: 报告 mean ± std)
================================================================
扫描当前目录下 result_*.json，按 (model, dataset, tile) 分组，
对 PSNR/SSIM/FPS 计算 mean ± std，输出可直接贴进论文的表格 (markdown + LaTeX)。

用法:  python aggregate_seeds.py [--glob 'result_*.json']
"""
import glob, json, argparse
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='result_*.json')
    args = ap.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for fn in sorted(glob.glob(args.glob)):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        key = (d.get('model', '?'), d.get('dataset', '?'), d.get('tile', 0))
        for m in ('psnr', 'ssim', 'fps'):
            if m in d and d[m] is not None:
                groups[key][m].append(d[m])
        if 'seed' in d:
            seeds[key].add(d['seed'])

    if not groups:
        print(f'No files matched {args.glob}'); return

    print('\n=== Markdown ===')
    print('| Model | Dataset | Infer | #Seeds | PSNR (mean±std) | SSIM (mean±std) | FPS |')
    print('|---|---|---|---|---|---|---|')
    latex = []
    for (model, ds, tile), mets in sorted(groups.items()):
        infer = 'full' if tile in (0, None) else f'tile{tile}'
        ns = len(seeds[(model, ds, tile)]) or len(mets.get('psnr', []))
        def ms(name, fmt):
            v = np.array(mets.get(name, []), dtype=float)
            if v.size == 0: return '-'
            return f'{v.mean():{fmt}}±{v.std(ddof=0):{fmt}}'
        psnr_s = ms('psnr', '.2f'); ssim_s = ms('ssim', '.4f'); fps_s = ms('fps', '.1f')
        print(f'| {model} | {ds} | {infer} | {ns} | {psnr_s} | {ssim_s} | {fps_s} |')
        latex.append(f'{model} & {ds} & {infer} & {psnr_s} & {ssim_s} \\\\')

    print('\n=== LaTeX rows ===')
    print('\n'.join(latex))


if __name__ == '__main__':
    main()
