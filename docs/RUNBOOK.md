# RUNBOOK — 返修实验执行手册
Submission 383164f3 · Deadline 2026-06-08 · 模型对外名建议: **HSCA-Net**(Hierarchical Spatial–Channel Attention)

> 本手册配合新增脚本：`model_arch_v2.py / train_v2.py / eval_v2.py / speed_bench_v2.py /
> aggregate_seeds.py / nhhaze_region_analysis.py`。原 `model_arch.py / eval_exp1.py` 保留不动，
> 作为「原文已发表结果」的复现基线。所有命令在你的 GPU 机器上运行；本助手环境无 GPU，
> 未产生任何实验数值。

---
## 0. 跑实验前必做的 3 件事（10 分钟）

1. **自检新模型**（确认与正文 1.34M / 0.32G@64 一致 + 任意分辨率可前向）：
   ```bash
   python model_arch_v2.py
   ```
   预期：`hier(正文)` 参数量≈1.34M、FLOPs@64≈0.32G、256/512 整图前向 OK；
   `no-hier` 参数量与之相近（仅 stride 改动，量级一致）。
   ⚠️ 若参数量与正文有出入，把 `model_arch_v2.UTNetV2` 默认 `channel/n_stages` 调到与原
   `UCTNet_SA_CA` 一致后再继续。

2. **确认数据目录结构**（dataset.py / eval_v2.py 的配对逻辑依赖它）：
   ```
   Haze4K/train/haze/*.png   Haze4K/train/gt/*.png
   Haze4K/test/haze/*.png    Haze4K/test/gt/*.png
   NH-HAZE/  *_hazy.png + *_GT.png   (同目录配对)
   ```

3. **记录硬件**（编辑 #7 要求真实硬件）：GPU 型号、CUDA、PyTorch 版本，写进论文实验设置。

---
## 🔴 P1-A 核心消融：有无分层下采样（最重要，先跑）

> 目的：直接证明"分层下采样→高效全局感受野"这一核心 claim。
> 对照组 `sa_ca_nohier` 与正文模型**深度/通道/block/多尺度监督完全相同**，唯一区别是
> 下采样被换成 stride=1（空间恒定），瓶颈窗口因此永远只覆盖 8×8 局部 → 无全局感受野。

```bash
# 正文模型(分层) ×3 seed
for s in 42 1 2; do
  python train_v2.py --model sa_ca --data_dir DATA/Haze4K/train \
    --save_dir runs/sa_ca_s$s --augment gamma --epochs 500 --seed $s
done
# 消融模型(无分层) ×3 seed
for s in 42 1 2; do
  python train_v2.py --model sa_ca_nohier --data_dir DATA/Haze4K/train \
    --save_dir runs/nohier_s$s --augment gamma --epochs 500 --seed $s
done
# 评测(整图前向，真正检验全局感受野)
for s in 42 1 2; do
  python eval_v2.py --model sa_ca        --ckpt runs/sa_ca_s$s/best.pth  --dataset haze4k --root DATA/Haze4K/test --seed $s
  python eval_v2.py --model sa_ca_nohier --ckpt runs/nohier_s$s/best.pth --dataset haze4k --root DATA/Haze4K/test --seed $s
done
python aggregate_seeds.py   # → 直接得到 mean±std 表
```
**预期论文论断**：分层版 PSNR 显著高于无分层版，且 FLOPs 更低（无分层在全分辨率做注意力更贵）。
若结果不显著，说明 claim 需弱化——这正是要先跑它的原因。

## 🔴 P1-B 多种子统计（编辑 #3）
上面已内置 3 seed。`aggregate_seeds.py` 输出 markdown+LaTeX 的 mean±std。
论文里报 `PSNR=xx.xx±0.0x`，并在正文写明 seeds={42,1,2}、重复 3 次。

## 🔴 P1-C 新增 Transformer baseline（编辑 #2）
**策略（重要，基于检索结论）**：DehazeFormer / MB-TaylorFormer / Restormer 原文**未在 Haze4K 报点**，
因此**必须在 Haze4K 上按统一协议重训**，不能挪用原文数字。流程：

1. 克隆官方 repo，仿照现有 `baseline_wrappers.py`（FFA/C2PNet 已就绪）写 wrapper：
   `__init__(in_channel=3, channel=32)` + `forward(x)->单张输出`，再用 `--loss_w 1`(仅主输出)训练。
   - DehazeFormer: https://github.com/IDKiro/DehazeFormer （用 dehazeformer-s/-b 轻量档对标）
   - Restormer:   https://github.com/swz30/Restormer
   - MB-TaylorFormer: https://github.com/FVL2020/MB-TaylorFormer
2. 统一协议：同 Haze4K/train、同 epochs、同 patch、同评测脚本 `eval_v2.py`（整图前向）。
3. **跑不通的 fallback**：在回复信中透明说明，并改用「published-results 对照表」(下方模板)，
   仅填写**确有 Haze4K 官方数字**的方法（如 C2PNet/PMNet 在 Haze4K≈33.49dB 量级，需回原文核对页码后引用）。
   ❗禁止为没有 Haze4K 数字的方法编造数值。

| Method | Venue | Params | Haze4K PSNR/SSIM | 来源 |
|---|---|---|---|---|
| FFA-Net | AAAI'20 | — | (你重训/或原文) | 重训:`runs/ffa_*` |
| C2PNet | CVPR'23 | — | (原文有,核对引用) | 原文 |
| DehazeFormer-s | TIP'23 | — | **重训填** | 统一协议 |
| Restormer | CVPR'22 | — | **重训填** | 统一协议 |
| MB-TaylorFormer | ICCV'23 | — | **重训填** | 统一协议 |
| HSCA-Net(ours) | — | 1.34M | **本实验填** | `eval_v2` |

---
## 🟠 P2 实验（P1 跑完再做）

**stage 数 / 窗口 / 头数消融**（编辑 #4，模型名即配置）：
```bash
python train_v2.py --model sa_ca_st2 ...   # 2 次下采样
python train_v2.py --model sa_ca_st4 ...   # 4 次下采样
python train_v2.py --model sa_ca_ws4 ...   # 窗口=4
python train_v2.py --model sa_ca_ws16 ...  # 窗口=16
python train_v2.py --model sa_ca_hd8 ...   # 8 头
```
**瓶颈分辨率消融**：等价于改 patch_size + n_stages 组合（瓶颈=patch/2^n_stages），在论文里用一张表呈现。

**大分辨率速度（编辑 #5/#7，回应"64×64 太小"）**：
```bash
python speed_bench_v2.py --model sa_ca        --sizes 64 256 512 1024
python speed_bench_v2.py --model sa_ca_nohier --sizes 64 256 512 1024
```
→ 得到 64/256/512/1024 各自 Params/FLOPs/FPS/latency 整图前向表。这张表把"全局感受野"卖点
落到真实分辨率上，是反驳 desk-reject 的关键证据。

**NH-Haze 分区域误差（编辑 #6）**：
```bash
python eval_v2.py --model sa_ca --ckpt runs/sa_ca_s42/best.pth --dataset nhhaze \
    --root DATA/NH-HAZE --save_dir results/nhhaze_pred
python nhhaze_region_analysis.py --root DATA/NH-HAZE --pred_dir results/nhhaze_pred \
    --heatmap_dir results/nhhaze_heatmaps
```
→ 输出 sky/dense_haze/vegetation/low_texture/high_texture 各区 PSNR，支撑"误差主要集中在 X"。

---
## 🟡 P3（有余力再做）
gUNet/PromptIR 额外轻量 baseline；SOTS/Dense-Haze/O-HAZE 跨域；Jetson 或代理推理；
复现编辑推荐文献中 1 篇(TransDehaze 或 UTMCR)作直接对比。

---
## 回复信必答 3 问（骨架，文本你来写）

**Q1 可复用性**：是。本文不止给出一个模型，而是提炼出**可复用设计原则**——
"分层下采样为轻量复原提供高效全局上下文"，并由 P1-A 消融定量验证；同时开源代码库 + 统一评测协议(`eval_v2.py` 整图前向) + NH-Haze 分区域误差分析协议，供他人引用/复现。

**Q2 可发现性**：是。新标题/摘要(编辑已给) 去模型名、突出 Hierarchical Spatial–Channel Attention；
关键词加入 lightweight restoration / window attention / global context；贡献列表改写为"可复用洞见"；
对比表扩到 5+ 方法含 Transformer baseline；图表加放大框 + 分区域误差。

**Q3 开源**：是。GitHub + Zenodo DOI，含 README/requirements/训练测试脚本/预训练权重/
复现主表与消融的**精确命令**(即本 RUNBOOK)，并注明关联本次 TVC 投稿。

---
## 时间线建议（10 天）
- D1–2: 跑 P1-A(核心消融) + P1-B(多种子)；同时起 baseline 重训(长任务后台挂)。
- D3–5: P1-C baseline 收尾 + P2 大分辨率速度 + NH-Haze 分区域。
- D5–7: P2 各消融补齐；整理表格(aggregate_seeds 直接出)。
- D7–9: 文本重写(标题/摘要/related work/方法/讨论) + 回复信 + 开源整理(README+DOI)。
- D10: 通读、查 \(8\times8\) 窗口等价性论证已推广到任意分辨率、提交。
```
