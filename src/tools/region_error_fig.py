"""
region_error_fig.py — 单张 NH-Haze 图的系统性分区域误差分析 + 配图 (编辑 #6)
==============================================================================
针对指定的一张 NH-Haze 测试图(默认 35), 把误差按编辑点名的 5 类区域系统分析:
  sky                  天空        (上部 + 高亮 + 低饱和)
  depth_discontinuity  深度突变    (强梯度边界, 作 depth discontinuity 的代理)
  vegetation           植被        (绿色通道占优)
  low_texture          低纹理      (局部梯度低, 平滑区)
  dense_haze           稠密雾      (hazy 相对 GT 被雾洗白最强的区域)

NH-Haze 无深度/语义标注, 故以上为无监督启发式 mask; 深度突变以强边缘近似,
论文需说明 "因 NH-Haze 无深度标注, 以强梯度边界近似深度不连续处"。

产出 (统一色标, 高 DPI, 适合直接进论文):
  panel.png        : 输入 / GT / 去雾 / 误差热力图  四联图(共享色条)
  region_map.png   : 5 类区域彩色掩膜叠加在 GT 上 + 图例
  region_psnr.png  : 该图逐区域 PSNR 柱状图
  region_<name>_err.png : 每类区域内的误差(其余置灰), 看误差是否集中
  region_error_<name>.json : 逐区域 PSNR / 平均误差数值(四位小数)

用法:
  python region_error_fig.py --root E:\\...\\NH-HAZE_split\\test ^
      --pred_dir results\\nhhaze_ours_joint --name 35 --out results\\region_fig_35
"""
import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# 5 类区域固定配色(论文统一色方案)
REGION_ORDER = ["sky", "depth_discontinuity", "vegetation", "low_texture", "dense_haze"]
REGION_CN = {
    "sky": "Sky",
    "depth_discontinuity": "Depth discontinuity",
    "vegetation": "Vegetation",
    "low_texture": "Low-texture",
    "dense_haze": "Dense haze",
}
REGION_COLOR = {
    "sky": (0.20, 0.55, 0.95),
    "depth_discontinuity": (0.95, 0.35, 0.20),
    "vegetation": (0.20, 0.75, 0.30),
    "low_texture": (0.75, 0.75, 0.20),
    "dense_haze": (0.70, 0.30, 0.85),
}


def load01(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


def grad_mag(gray):
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    return np.sqrt(gx ** 2 + gy ** 2)


def region_masks(hazy, gt):
    h, w, _ = gt.shape
    bright = gt.mean(2)
    mx = gt.max(2); mn = gt.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    yy = np.linspace(0, 1, h)[:, None].repeat(w, 1)
    sky = (yy < 0.45) & (bright > 0.6) & (sat < 0.25)
    haze_strength = hazy.mean(2) - gt.mean(2)
    dense = haze_strength >= np.quantile(haze_strength, 0.75)
    veg = (gt[..., 1] > gt[..., 0] + 0.04) & (gt[..., 1] > gt[..., 2] + 0.04)
    g = grad_mag(gt.mean(2))
    low_tex = g <= np.quantile(g, 0.30)
    depth_disc = g >= np.quantile(g, 0.85)   # 强边缘 ~ 深度不连续代理
    return {"sky": sky, "depth_discontinuity": depth_disc, "vegetation": veg,
            "low_texture": low_tex, "dense_haze": dense}


def masked_psnr(pred, gt, mask, eps=1e-10):
    if mask.sum() < 50:
        return None
    m = mask[..., None]
    mse = ((pred - gt) ** 2 * m).sum() / (m.sum() * 3 + eps)
    return float(10.0 * np.log10(1.0 / max(mse, eps)))


def masked_mae(pred, gt, mask, eps=1e-10):
    if mask.sum() < 50:
        return None
    m = mask[..., None]
    return float((np.abs(pred - gt) * m).sum() / (m.sum() * 3 + eps))


def find_one(root, pred_dir, name):
    """返回 (hazy_path, gt_path, pred_path)。兼容 haze/gt 子目录 或 *_hazy/_GT。"""
    root = os.path.abspath(root)
    hz_dir, gt_dir = os.path.join(root, "haze"), os.path.join(root, "gt")
    hazy = gtp = None
    if os.path.isdir(hz_dir) and os.path.isdir(gt_dir):
        for e in IMG_EXTS:
            if os.path.exists(os.path.join(hz_dir, name + e)):
                hazy = os.path.join(hz_dir, name + e); break
        for e in IMG_EXTS:
            if os.path.exists(os.path.join(gt_dir, name + e)):
                gtp = os.path.join(gt_dir, name + e); break
    if hazy is None:  # 结构2
        for f in os.listdir(root):
            s = os.path.splitext(f)[0]
            if s.lower().startswith(name.lower()) and "_hazy" in s.lower():
                hazy = os.path.join(root, f)
                gtp = os.path.join(root, f.replace("_hazy", "_GT").replace("_Hazy", "_GT"))
                break
    # pred
    predp = None
    for e in IMG_EXTS:
        if os.path.exists(os.path.join(pred_dir, name + e)):
            predp = os.path.join(pred_dir, name + e); break
    if predp is None:
        for f in os.listdir(pred_dir):
            if os.path.splitext(f)[0].split("_")[0] == name:
                predp = os.path.join(pred_dir, f); break
    return hazy, gtp, predp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="NH-Haze test 根目录(含 haze/gt 或 *_hazy/_GT)")
    ap.add_argument("--pred_dir", required=True, help="eval_v2 --save_dir 的去雾结果目录")
    ap.add_argument("--name", default="35", help="图名(不含扩展名), 默认 35")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--cmap", default="turbo", help="误差热力图色标(turbo/jet/viridis)")
    ap.add_argument("--title", default="UTNet", help="四联图里去雾结果那张的方法名标题")
    args = ap.parse_args()
    out = args.out or os.path.join("results", f"region_fig_{args.name}")
    os.makedirs(out, exist_ok=True)

    hazy_p, gt_p, pred_p = find_one(args.root, args.pred_dir, args.name)
    if not (hazy_p and gt_p and pred_p):
        print(f"ERROR: 找不到齐全的三张图 name={args.name}")
        print(f"  hazy={hazy_p}\n  gt  ={gt_p}\n  pred={pred_p}")
        return
    hazy, gt, pred = load01(hazy_p), load01(gt_p), load01(pred_p)
    H = min(gt.shape[0], pred.shape[0], hazy.shape[0])
    W = min(gt.shape[1], pred.shape[1], hazy.shape[1])
    hazy, gt, pred = hazy[:H, :W], gt[:H, :W], pred[:H, :W]

    err = ((pred - gt) ** 2).mean(2)          # 逐像素 MSE
    masks = region_masks(hazy, gt)

    # ---- 逐区域数值 ----
    rows = []
    overall_psnr = masked_psnr(pred, gt, np.ones((H, W), bool))
    overall_mae = masked_mae(pred, gt, np.ones((H, W), bool))
    for name in REGION_ORDER:
        m = masks[name]
        rows.append({
            "region": name,
            "psnr": round(masked_psnr(pred, gt, m) or float("nan"), 4),
            "mae": round(masked_mae(pred, gt, m) or float("nan"), 4),
            "coverage": round(float(m.mean()), 4),
        })
    summary = {"name": args.name, "overall_psnr": round(overall_psnr, 4),
               "overall_mae": round(overall_mae, 4), "regions": rows}
    json.dump(summary, open(os.path.join(out, f"region_error_{args.name}.json"),
                            "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n=== NH-Haze {args.name} 分区域误差 (PSNR 越低=误差越大) ===")
    print(f'{"region":<22}{"PSNR":>10}{"MAE":>10}{"coverage":>10}')
    print(f'{"OVERALL":<22}{overall_psnr:>10.4f}{overall_mae:>10.4f}{1.0:>10.4f}')
    for r in rows:
        print(f'{r["region"]:<22}{r["psnr"]:>10.4f}{r["mae"]:>10.4f}{r["coverage"]:>10.4f}')

    vmax = float(np.quantile(err, 0.99))      # 统一色标上限(去极端值)

    # ---- 图1: 四联面板 输入/GT/去雾/误差 ----
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, im, t in zip(ax[:3], [hazy, gt, pred], ["Hazy input", "Ground truth", args.title]):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=12); a.axis("off")
    im3 = ax[3].imshow(err, cmap=args.cmap, vmin=0, vmax=vmax)
    ax[3].set_title("Per-pixel error (MSE)", fontsize=12); ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panel.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 区域彩色掩膜叠加 ----
    overlay = gt.copy()
    for name in REGION_ORDER:
        c = np.array(REGION_COLOR[name])
        m = masks[name][..., None]
        overlay = overlay * (1 - 0.45 * m) + 0.45 * m * c
    fig, a = plt.subplots(figsize=(7, 6))
    a.imshow(np.clip(overlay, 0, 1)); a.axis("off")
    a.set_title(f"Region partition (NH-Haze {args.name})", fontsize=13)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
               markerfacecolor=REGION_COLOR[n], markeredgecolor="k",
               label=f"{REGION_CN[n]} ({masks[n].mean()*100:.1f}%)") for n in REGION_ORDER]
    a.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
             ncol=3, fontsize=9, frameon=False)
    fig.savefig(os.path.join(out, "region_map.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 逐区域 PSNR 柱状图 ----
    fig, a = plt.subplots(figsize=(7, 4))
    names = [REGION_CN[r["region"]] for r in rows]
    vals = [r["psnr"] for r in rows]
    cols = [REGION_COLOR[r["region"]] for r in rows]
    bars = a.bar(names, vals, color=cols, edgecolor="k")
    a.axhline(overall_psnr, color="gray", ls="--", lw=1, label=f"Overall {overall_psnr:.2f} dB")
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    a.set_ylabel("Region PSNR (dB)"); a.set_title(f"Per-region PSNR (NH-Haze {args.name})")
    a.legend(fontsize=9); plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "region_psnr.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 每类区域内误差(其余置灰), 看误差集中 ----
    for name in REGION_ORDER:
        m = masks[name]
        masked_err = np.where(m, err, np.nan)
        fig, a = plt.subplots(figsize=(6, 5))
        a.imshow(gt * 0.3 + 0.2)  # 暗背景
        im = a.imshow(masked_err, cmap=args.cmap, vmin=0, vmax=vmax)
        a.set_title(f"{REGION_CN[name]} error", fontsize=12); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(out, f"region_{name}_err.png"),
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved figures + json -> {out}")


if __name__ == "__main__":
    main()
