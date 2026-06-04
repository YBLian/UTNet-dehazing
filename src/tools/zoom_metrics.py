"""
zoom_metrics.py — 放大框区域的 PSNR / SSIM 量化 (配合 make_zoom.py)
====================================================================
读取【未画框的原图】(results/paper_figs) + make_zoom.py 产出的 zoom_boxes.json,
对每个场景、每个方法, 裁出与放大图【完全相同】的红框区域, 与 GT 同位置区域
比较, 计算区域 PSNR / SSIM。输出 markdown 表 + json。

为什么读 paper_figs 而不是 paper_figs_zoom:
  带框图上画了红框、又拼了放大块, 像素被污染, 不能用来算指标。
  必须用原始去雾结果(paper_figs)按相同坐标裁区域。

SSIM 实现与 eval_v2.py 一致(高斯窗 11, sigma 1.5, C1=0.01^2, C2=0.03^2, 逐通道均值),
保证和论文主表口径统一。仅依赖 numpy + Pillow。

用法:
  python zoom_metrics.py
  python zoom_metrics.py --src results\\paper_figs --boxes results\\paper_figs_zoom\\zoom_boxes.json
"""
import os
import json
import argparse
import numpy as np
from PIL import Image

# 这些不是"方法"输出, 不参与打分
SKIP = {"hazy", "GT"}
METHOD_ORDER = ["ours", "ours_zeroshot", "ours_joint", "DF-S", "DF-T"]


def load01(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def resize_to(arr, size_wh):
    """arr: HxWx3 [0,1]; size_wh=(W,H)。用双线性把 arr 缩放到目标尺寸。"""
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize(size_wh, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def crop(arr, box):
    x, y, w, h = box
    return arr[y:y + h, x:x + w, :]


def psnr(pred, gt, eps=1e-10):
    mse = np.mean((pred - gt) ** 2)
    mse = max(mse, eps)
    return float(10.0 * np.log10(1.0 / mse))


def _gauss1d(ws=11, sigma=1.5):
    c = np.arange(ws) - ws // 2
    g = np.exp(-(c ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _sepconv(x, k):
    """对 HxW 做可分离高斯卷积(reflect padding, 'same')。"""
    ws = len(k)
    p = ws // 2
    xp = np.pad(x, ((p, p), (0, 0)), mode="reflect")
    tmp = np.zeros_like(x)
    for i in range(ws):
        tmp += k[i] * xp[i:i + x.shape[0], :]
    xp = np.pad(tmp, ((0, 0), (p, p)), mode="reflect")
    out = np.zeros_like(x)
    for i in range(ws):
        out += k[i] * xp[:, i:i + x.shape[1]]
    return out


def ssim(pred, gt, ws=11, sigma=1.5):
    """逐通道高斯窗 SSIM, 取通道均值; 输入 HxWx3 [0,1]。"""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = _gauss1d(ws, sigma)
    vals = []
    for ch in range(pred.shape[2]):
        x = pred[..., ch]
        y = gt[..., ch]
        mx = _sepconv(x, k)
        my = _sepconv(y, k)
        sx = _sepconv(x * x, k) - mx * mx
        sy = _sepconv(y * y, k) - my * my
        sxy = _sepconv(x * y, k) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / \
            ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))
        vals.append(float(m.mean()))
    return float(np.mean(vals))


def find_file(folder, key):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(folder, key + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("results", "paper_figs"))
    ap.add_argument("--boxes",
                    default=os.path.join("results", "paper_figs_zoom", "zoom_boxes.json"))
    ap.add_argument("--out", default="zoom_metrics.json")
    args = ap.parse_args()

    if not os.path.exists(args.boxes):
        print(f"ERROR: 找不到 {args.boxes}; 请先运行 make_zoom.py 生成放大图(会一并写出 zoom_boxes.json)")
        return
    boxes = json.load(open(args.boxes, "r", encoding="utf-8"))

    results = {}
    print("\n=== 放大区域 PSNR / SSIM (region-only, vs GT) ===")
    for scene in sorted(boxes):
        sdir = os.path.join(args.src, scene)
        gtp = find_file(sdir, "GT")
        if gtp is None:
            print(f"[SKIP] {scene}: 无 GT"); continue
        gt = load01(gtp)
        H, W = gt.shape[:2]
        box = boxes[scene]
        gt_c = crop(gt, box)
        if gt_c.size == 0:
            print(f"[SKIP] {scene}: box 越界 {box} vs GT {W}x{H}"); continue

        present = [k for k in METHOD_ORDER if find_file(sdir, k)]
        present += [os.path.splitext(f)[0] for f in os.listdir(sdir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    and os.path.splitext(f)[0] not in SKIP
                    and os.path.splitext(f)[0] not in present]
        results[scene] = {}
        print(f"\n[{scene}]  box={box}")
        print(f"  {'method':<14}{'PSNR':>10}{'SSIM':>10}")
        for k in present:
            arr = load01(find_file(sdir, k))
            if arr.shape[:2] != (H, W):
                arr = resize_to(arr, (W, H))
            pc = crop(arr, box)
            ps = psnr(pc, gt_c); ss = ssim(pc, gt_c)
            results[scene][k] = {"psnr": round(ps, 4), "ssim": round(ss, 4)}
            print(f"  {k:<14}{ps:>10.4f}{ss:>10.4f}")

    # 分组平均(haze4k_* 一组, nhhaze_* 一组), 便于写"放大区平均"
    def group_mean(prefix):
        agg = {}
        for scene, md in results.items():
            if not scene.startswith(prefix):
                continue
            for k, v in md.items():
                agg.setdefault(k, {"psnr": [], "ssim": []})
                agg[k]["psnr"].append(v["psnr"]); agg[k]["ssim"].append(v["ssim"])
        return {k: {"psnr": round(float(np.mean(v["psnr"])), 4),
                    "ssim": round(float(np.mean(v["ssim"])), 4),
                    "n": len(v["psnr"])} for k, v in agg.items()}

    summary = {"haze4k_mean": group_mean("haze4k"),
               "nhhaze_mean": group_mean("nhhaze")}
    for grp, gm in summary.items():
        if not gm:
            continue
        print(f"\n=== {grp} (放大区平均) ===")
        print(f"  {'method':<14}{'PSNR':>10}{'SSIM':>10}{'#scenes':>9}")
        for k in [m for m in METHOD_ORDER if m in gm] + [m for m in gm if m not in METHOD_ORDER]:
            print(f"  {k:<14}{gm[k]['psnr']:>10.4f}{gm[k]['ssim']:>10.4f}{gm[k]['n']:>9}")

    json.dump({"per_scene": results, "summary": summary},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
