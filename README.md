# UTNet-dehazing

**Lightweight U-shaped Transformer with hierarchical spatial–channel attention for single-image dehazing (The Visual Computer).**

UTNet is a lightweight U-shaped Transformer for single-image dehazing. It obtains a global receptive field through a hierarchical downsampling architecture (rather than heavier or shifted-window attention), and combines local-window spatial attention with parallel channel attention (HSCBlock) plus multi-scale auxiliary supervision and gamma-based augmentation. UTNet reaches competitive restoration quality with only **1.34M parameters** and **0.32G FLOPs**.

> This repository is associated with our manuscript submitted to *The Visual Computer*. See the **Code Availability Statement** below.

---

## Code Availability Statement

To support reproducibility and reuse, the complete source code, documentation, and experimental scripts of UTNet are publicly available on GitHub at `https://github.com/LianYB/UTNet-dehazing`, and a permanently archived snapshot is deposited on Zenodo with DOI `https://doi.org/10.5281/zenodo.<ZENODO_ID>`. The repository provides: (i) full documentation (this README and a step-by-step runbook); (ii) the environment configuration (`requirements.txt`, with the tested setup of Python 3.10, PyTorch 2.9.0 and CUDA 12.8); (iii) training and testing scripts for UTNet and the compared baselines; (iv) pretrained model weights; and (v) the exact commands required to reproduce the main quantitative tables, the ablation studies, and the qualitative analyses reported in the paper, including the multi-resolution complexity benchmark, the hierarchical-downsampling and window-size ablations, the NH-Haze cross-domain evaluation, the region-wise error analysis, and the zoomed-in visual comparisons. This repository is associated with the present manuscript submitted to *The Visual Computer* and may be cited together with the article. The archived Zenodo record provides a citable, version-fixed reference with a persistent DOI.

> Replace `<ZENODO_ID>` with the DOI generated after you create a tagged release (e.g. `v1.0`) and enable Zenodo archiving for this repository.

---

## Environment

Tested on a single NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB), Python 3.10, PyTorch 2.9.0, CUDA 12.8.

```bash
# 1) install PyTorch matching your CUDA (see https://pytorch.org)
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) install the rest
pip install -r requirements.txt
```

## Datasets

The datasets are **not** included in this repository. Download them from the official sources and arrange as below.

- **Haze4K** — From synthetic to real: image dehazing collaborating with unlabeled real data (ACM MM 2021).
- **NH-Haze** — NH-HAZE: a non-homogeneous image dehazing benchmark (CVPRW 2020).

Expected directory layout:

```
data/
  Haze4K/
    train/        # 3000 hazy/clean pairs
    test/         # 1000 hazy/clean pairs
  NH-HAZE_split/  # produced by split_nhhaze.py (8:2 split, fixed 13-image test set)
    train/
      haze/  gt/
    test/
      haze/  gt/
```

To create the NH-Haze 8:2 split used in the paper (fixed 13-image test set):

```bash
python split_nhhaze.py --root data/NH-HAZE --out data/NH-HAZE_split --ratio 0.8
```

## Repository structure

```
UTNet-dehazing/
  README.md
  requirements.txt
  .gitignore
  src/                       # model + training/eval scripts
    model_arch_v2.py         # UTNet architecture
    build_any.py             # model factory (UTNet / DehazeFormer / ...)
    train_v2.py              # training
    eval_v2.py               # evaluation (whole-image + tiled inference)
    dataset.py               # data loading
    multi_dataset.py         # joint (Haze4K + NH-Haze) data loading
    split_nhhaze.py          # NH-Haze 8:2 split
    speed_bench_v2.py        # FLOPs / FPS / latency benchmark
    aggregate_seeds.py       # mean +/- std over seeds
    region_error_fig.py      # region-wise error analysis + figures (Sky/Depth/Veg/Low-tex/Dense)
    make_zoom.py             # zoomed-in crops for visual comparison
    zoom_metrics.py          # metrics on zoomed regions
    dehazeformer_wrapper.py  # DehazeFormer baseline wrapper
  docs/
    RUNBOOK.md               # step-by-step reproduction notes
```

## Pretrained weights

Weights are released as **GitHub Release assets** / on **Zenodo** (not tracked in git). Download and place under `runs/` as referenced by the commands below. Key checkpoints:

| Checkpoint | Description |
|---|---|
| `ours_e500/best.pth` | UTNet main model (Haze4K, 500 epochs) |
| `dfs_e300/best.pth` | DehazeFormer-S (Haze4K retrained) |
| `dft_e300/best.pth` | DehazeFormer-T (Haze4K retrained) |
| `joint_e300_v2/best.pth` | UTNet joint training (Haze4K + NH-Haze) |

## Reproduce the paper

> Paths below are placeholders — adjust to your local dataset locations.

### Main quantitative table & efficiency

```bash
# UTNet (main) + DehazeFormer baselines on Haze4K
python eval_v2.py --model sa_ca          --ckpt runs/ours_e500/best.pth --dataset haze4k --root data/Haze4K/test --seed 42
python eval_v2.py --model dehazeformer_s --ckpt runs/dfs_e300/best.pth  --dataset haze4k --root data/Haze4K/test --seed 42
python eval_v2.py --model dehazeformer_t --ckpt runs/dft_e300/best.pth  --dataset haze4k --root data/Haze4K/test --seed 42

# Complexity / speed (Params, FLOPs, FPS, Latency) at multiple resolutions
python speed_bench_v2.py --model sa_ca --sizes 64 256 512 1024
```

### Ablations

```bash
# Encoder/decoder depth (number of downsampling stages)
python eval_v2.py --model sa_ca_st2 --ckpt runs/st2_e300/best.pth --dataset haze4k --root data/Haze4K/test

# Window size
python eval_v2.py --model sa_ca_ws4 --ckpt runs/ws4_e300/best.pth --dataset haze4k --root data/Haze4K/test

# Hierarchical downsampling (3 seeds -> mean +/- std)
python eval_v2.py --model sa_ca        --ckpt runs/hier30_s42/best.pth  --dataset haze4k --root data/Haze4K/test --seed 42
python eval_v2.py --model sa_ca_nohier --ckpt runs/nohier_s42/best.pth  --dataset haze4k --root data/Haze4K/test --seed 42
python aggregate_seeds.py --glob "result_sa_ca_haze4k_seed*.json"
```

### NH-Haze cross-domain (zero-shot) & qualitative analysis

```bash
# Zero-shot: UTNet trained on Haze4K only, evaluated on NH-Haze (13-image test set)
python eval_v2.py --model sa_ca --ckpt runs/ours_e500/best.pth --dataset nhhaze --root data/NH-HAZE_split/test --save_dir results/nhhaze_ours_zeroshot --seed 42

# Region-wise error analysis (Sky / Depth discontinuity / Vegetation / Low-texture / Dense haze) on image No. 35
python region_error_fig.py --root data/NH-HAZE_split/test --pred_dir results/nhhaze_ours_zeroshot --name 35 --title UTNet

# Zoomed-in visual comparison crops + per-crop metrics
python make_zoom.py --src results/paper_figs --out results/paper_figs_zoom
python zoom_metrics.py
```

### NH-Haze joint-training experiment (this experiment only)

```bash
# Train UTNet jointly on Haze4K + NH-Haze training split
python train_v2.py --model sa_ca --data_dir data/Haze4K/train \
  --extra_data data/NH-HAZE_split/train --extra_repeat 30 \
  --save_dir runs/joint_e300_v2 --epochs 300

# Evaluate the joint model
python eval_v2.py --model sa_ca --ckpt runs/joint_e300_v2/best.pth --dataset nhhaze --root data/NH-HAZE_split/test --seed 42
```

> Note: the joint-training model is used **only** for the NH-Haze training experiment. All other results (main table, efficiency, ablations, error analysis, visualizations) use the Haze4K-only model.

## Citation

```bibtex
@article{utnet_dehazing,
  title   = {A Lightweight U-Shaped Transformer Network for Single Image Dehazing},
  author  = {Zhang, Yiping and Lian, Youbo},
  journal = {The Visual Computer},
  note    = {Code: https://github.com/LianYB/UTNet-dehazing},
  year    = {2026}
}
```
