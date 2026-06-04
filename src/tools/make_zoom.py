"""
make_zoom.py — 论文用局部放大对比图生成器 (编辑 #6: zoomed-in regions)
======================================================================
对 results/paper_figs 下每个场景文件夹里的所有方法图, 在【同一坐标】裁一个
放大框, 在原图上画红框并把放大块拼到下方(或右侧), 输出到 results/paper_figs_zoom。

关键保证: 同一场景的 hazy / GT / 各方法去雾结果使用【完全相同】的框坐标,
          否则方法间对比不公平。

选框方式:
  默认  : 按 GT 的局部梯度自动选"细节最丰富"的区域(最能体现去雾差异)。
  手动  : --box x,y,w,h 指定坐标(配 --only 单独重切某个场景)。

用法:
  python make_zoom.py --src results\\paper_figs --out results\\paper_figs_zoom
  python make_zoom.py --only haze4k_273 --box 120,80,180,180   # 手动重切一张
  python make_zoom.py --layout right --bw 200 --bh 200          # 放大块放右边/改框大小
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

# 输出顺序(论文常用从左到右): 输入 -> 真值 -> 本文 -> 各 baseline
METHOD_ORDER = ["hazy", "GT", "ours", "ours_zeroshot", "ours_joint", "DF-S", "DF-T"]


def load(p):
    return Image.open(p).convert("RGB")


def grad_map(img):
    """灰度梯度幅值, 用于衡量局部细节丰富程度。"""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    gy = np.zeros_like(a)
    gx = np.zeros_like(a)
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    return np.sqrt(gx ** 2 + gy ** 2)


def auto_box(gt, bw, bh):
    """用积分图在 GT 上找梯度和最大的 bw x bh 窗口。"""
    g = grad_map(gt)
    H, W = g.shape
    bw = min(bw, W)
    bh = min(bh, H)
    ii = np.cumsum(np.cumsum(g, 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    best, bx, by = -1.0, 0, 0
    step = max(8, min(bw, bh) // 4)
    for y in range(0, H - bh + 1, step):
        for x in range(0, W - bw + 1, step):
            s = ii[y + bh, x + bw] - ii[y, x + bw] - ii[y + bh, x] + ii[y, x]
            if s > best:
                best, bx, by = s, x, y
    return bx, by, bw, bh


def draw_and_stack(img, box, line=4, layout="bottom"):
    """在原图画红框, 并把放大块拼到下方/右侧, 放大块也加红框。"""
    x, y, w, h = box
    W, H = img.size
    big = img.copy()
    d = ImageDraw.Draw(big)
    d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=line)
    crop = img.crop((x, y, x + w, y + h))
    if layout == "bottom":
        zh = max(1, int(round(W * h / w)))
        zc = crop.resize((W, zh))
        out = Image.new("RGB", (W, H + zh + line), (255, 255, 255))
        out.paste(big, (0, 0))
        out.paste(zc, (0, H + line))
        d2 = ImageDraw.Draw(out)
        d2.rectangle([0, H, W - 1, H + zh + line - 1], outline=(255, 0, 0), width=line)
    else:  # right
        zw = max(1, int(round(H * w / h)))
        zc = crop.resize((zw, H))
        out = Image.new("RGB", (W + zw + line, H), (255, 255, 255))
        out.paste(big, (0, 0))
        out.paste(zc, (W + line, 0))
        d2 = ImageDraw.Draw(out)
        d2.rectangle([W, 0, W + zw + line - 1, H - 1], outline=(255, 0, 0), width=line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("results", "paper_figs"))
    ap.add_argument("--out", default=os.path.join("results", "paper_figs_zoom"))
    ap.add_argument("--bw", type=int, default=0, help="框宽 (0=图宽的1/3)")
    ap.add_argument("--bh", type=int, default=0, help="框高 (0=图高的1/3)")
    ap.add_argument("--layout", choices=["bottom", "right"], default="bottom")
    ap.add_argument("--box", default="", help="手动 x,y,w,h (留空=按 --mode 选框)")
    ap.add_argument("--mode", choices=["auto", "center"], default="auto",
                    help="auto=按GT梯度选最密细节区; center=居中固定框(中立, 不偏向任何方法)")
    ap.add_argument("--only", default="", help="只处理某子文件夹, 如 haze4k_125")
    ap.add_argument("--prefix", default="", help="只处理名字以此开头的子文件夹, 如 haze4k")
    ap.add_argument("--line", type=int, default=4, help="红框线宽")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: 找不到源目录 {args.src}")
        return

    folders = [d for d in sorted(os.listdir(args.src))
               if os.path.isdir(os.path.join(args.src, d))
               and (not args.only or d == args.only)
               and (not args.prefix or d.startswith(args.prefix))]
    if not folders:
        print(f"ERROR: {args.src} 下没有匹配的子文件夹"
              + (f" (--only {args.only})" if args.only else ""))
        return

    # 读已有 box 记录(若存在), 使 --only 单独重切某场景时不覆盖其它场景坐标
    import json as _json
    os.makedirs(args.out, exist_ok=True)
    box_json = os.path.join(args.out, "zoom_boxes.json")
    boxes = {}
    if os.path.exists(box_json):
        try:
            boxes = _json.load(open(box_json, "r", encoding="utf-8"))
        except Exception:
            boxes = {}

    for fd in folders:
        sdir = os.path.join(args.src, fd)
        odir = os.path.join(args.out, fd)
        os.makedirs(odir, exist_ok=True)
        imgs = {os.path.splitext(f)[0]: os.path.join(sdir, f)
                for f in os.listdir(sdir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))}
        if not imgs:
            print(f"[SKIP] {fd}: 无图片")
            continue

        # 参考尺寸/选框基准: 优先 GT
        ref = load(imgs["GT"]) if "GT" in imgs else load(next(iter(imgs.values())))
        W, H = ref.size
        bw = args.bw or W // 3
        bh = args.bh or H // 3
        if args.box:
            x, y, w, h = [int(v) for v in args.box.split(",")]
            box = (x, y, w, h)
        elif args.mode == "center":
            bw = min(bw, W); bh = min(bh, H)
            box = ((W - bw) // 2, (H - bh) // 2, bw, bh)
        else:
            box = auto_box(ref, bw, bh)

        order = ([k for k in METHOD_ORDER if k in imgs]
                 + [k for k in imgs if k not in METHOD_ORDER])
        for k in order:
            im = load(imgs[k])
            if im.size != (W, H):
                im = im.resize((W, H))
            out = draw_and_stack(im, box, line=args.line, layout=args.layout)
            out.save(os.path.join(odir, k + ".png"))
        boxes[fd] = [int(v) for v in box]
        print(f"[OK] {fd}  box={box}  ({len(order)} imgs)")

    _json.dump(boxes, open(box_json, "w", encoding="utf-8"), indent=2)
    print(f"Done -> {args.out}  | boxes -> {box_json}")


if __name__ == "__main__":
    main()
