"""
multi_dataset.py — 联合数据集 (多数据根 + 过采样)
================================================
复用 SIRRDataset, 把多个数据根拼成一个训练集, 并对指定根做整数倍过采样。
用于问题3: Haze4K(3000) + NH-Haze train(~44, 过采样 30x≈1320) 联合训练。

ConcatOversampleDataset(roots, repeats, **dataset_kwargs)
  roots   : [root1, root2, ...] 每个根下需有 haze/ 和 gt/
  repeats : [r1, r2, ...] 与 roots 一一对应, 该根样本在一个 epoch 内重复次数(整数)
所有 dataset_kwargs (input_label/target_label/patch_size/augment/...) 透传给 SIRRDataset。
"""
from torch.utils.data import Dataset
from dataset import SIRRDataset


class ConcatOversampleDataset(Dataset):
    def __init__(self, roots, repeats, **dataset_kwargs):
        assert len(roots) == len(repeats), 'roots 与 repeats 数量需一致'
        self.subsets = []
        self.index_map = []  # [(subset_i, local_idx), ...]
        for si, (root, rep) in enumerate(zip(roots, repeats)):
            ds = SIRRDataset(root_dir=root, **dataset_kwargs)
            self.subsets.append(ds)
            n = len(ds)
            if n == 0:
                print(f'[MultiDataset] WARNING: 0 pairs in {root}')
            for _ in range(int(rep)):
                for li in range(n):
                    self.index_map.append((si, li))
        # 汇总打印
        summary = ', '.join(
            f'{root}×{rep}={len(ds)*int(rep)}'
            for root, rep, ds in zip(roots, repeats, self.subsets))
        print(f'[MultiDataset] total={len(self.index_map)} samples/epoch | {summary}')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        si, li = self.index_map[idx]
        return self.subsets[si][li]
