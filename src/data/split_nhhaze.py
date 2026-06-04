"""
split_nhhaze.py — NH-Haze 8:2 划分 (尾部 20% 作 test)
=====================================================
输入: 一个含 NN_hazy.png / NN_GT.png 配对的目录 (同目录)。
输出: 在 --out 下生成
    train/haze/*.png  train/gt/*.png   (前 80%, 按编号排序)
    test/haze/*.png   test/gt/*.png    (后 20%)
文件复制后统一改名为 gt 用 <id>.png、haze 用 <id>.png, 以匹配 dataset.py 的 haze/gt 配对逻辑。
划分名单写入 split_manifest.txt, 便于写进论文 / 开源复现。

用法:
  python split_nhhaze.py --src E:\\...\\data\\NH-HAZE --out E:\\...\\data\\NH-HAZE_split --ratio 0.8
"""
import os, re, shutil, argparse
from pathlib import Path

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def find_pairs(src):
    src = Path(src)
    hazy = {}
    gt = {}
    for p in src.rglob('*'):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        name = p.stem  # e.g. '01_hazy' or '01_GT'
        m = re.match(r'(.+?)_(hazy|GT)$', name, flags=re.IGNORECASE)
        if not m:
            continue
        idx, kind = m.group(1), m.group(2).lower()
        if kind == 'hazy':
            hazy[idx] = p
        else:
            gt[idx] = p
    ids = sorted(set(hazy) & set(gt), key=lambda s: (len(s), s))
    pairs = [(i, hazy[i], gt[i]) for i in ids]
    return pairs


def copy_split(pairs, out, split):
    hz = Path(out) / split / 'haze'
    gtd = Path(out) / split / 'gt'
    hz.mkdir(parents=True, exist_ok=True)
    gtd.mkdir(parents=True, exist_ok=True)
    for idx, hp, gp in pairs:
        shutil.copy2(hp, hz / f'{idx}.png')
        shutil.copy2(gp, gtd / f'{idx}.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ratio', type=float, default=0.8, help='train 比例 (尾部 1-ratio 作 test)')
    ap.add_argument('--force_test', default='', help='逗号分隔的编号, 强制进test, 并从原test尾部等量挪回train保持比例, 如 "35,43"')
    args = ap.parse_args()

    pairs = find_pairs(args.src)
    n = len(pairs)
    if n == 0:
        print('ERROR: 没找到 NN_hazy/NN_GT 配对'); return
    n_train = int(round(n * args.ratio))
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    # 强制指定编号进 test, 从原 test 尾部等量回填 train, 保持 train/test 数量不变
    force = [s.strip() for s in args.force_test.split(',') if s.strip()]
    if force:
        id2pair = {i: (i, hp, gp) for i, hp, gp in pairs}
        missing = [i for i in force if i not in id2pair]
        if missing:
            print(f'ERROR: force_test 编号不存在: {missing}'); return
        train_ids = [i for i, _, _ in train_pairs]
        test_ids = [i for i, _, _ in test_pairs]
        moved_in = []
        for fid in force:
            if fid in test_ids:
                continue  # 已在test, 跳过
            # 从 train 移除 fid, 加入 test; 从原 test 尾部取一个回填 train
            if fid in train_ids:
                train_ids.remove(fid)
                test_ids.append(fid)
                moved_in.append(fid)
        # 从 test 尾部(排除刚加入的force) 挪等量回 train, 保持数量
        need_back = len(moved_in)
        movable = [i for i in test_ids if i not in force]
        back = movable[-need_back:] if need_back else []
        for bid in back:
            test_ids.remove(bid)
            train_ids.insert(0, bid)
        # 重新按编号排序并重建 pairs
        train_ids = sorted(set(train_ids), key=lambda s: (len(s), s))
        test_ids = sorted(set(test_ids), key=lambda s: (len(s), s))
        train_pairs = [id2pair[i] for i in train_ids]
        test_pairs = [id2pair[i] for i in test_ids]
        print(f'[force_test] 强制进test: {moved_in} | 回填train: {back}')

    copy_split(train_pairs, args.out, 'train')
    copy_split(test_pairs, args.out, 'test')

    manifest = Path(args.out) / 'split_manifest.txt'
    with open(manifest, 'w') as f:
        f.write(f'NH-Haze split (ratio={args.ratio}, tail as test)\n')
        if args.force_test:
            f.write(f'force_test={args.force_test} (指定编号强制进test, 等量回填train)\n')
        f.write(f'total={n}  train={len(train_pairs)}  test={len(test_pairs)}\n\n')
        f.write('[TRAIN ids]\n' + ', '.join(i for i, _, _ in train_pairs) + '\n\n')
        f.write('[TEST ids]\n' + ', '.join(i for i, _, _ in test_pairs) + '\n')

    print(f'Total pairs: {n}')
    print(f'  Train: {len(train_pairs)} -> {Path(args.out)/"train"}')
    print(f'  Test : {len(test_pairs)} -> {Path(args.out)/"test"}')
    print(f'  Test ids: {[i for i,_,_ in test_pairs]}')
    print(f'Manifest -> {manifest}')


if __name__ == '__main__':
    main()
