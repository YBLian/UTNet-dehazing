"""
eval_v2.py — UTNet v2 整图(full-image)评测
============================================
默认整图前向（检验真实全局感受野）；可选 --tile 做滑窗对照。
输出 PSNR / SSIM / FPS 到 JSON，文件名含 model/dataset/seed，便于多种子聚合。

用法:
  python eval_v2.py --model sa_ca --ckpt runs/sa_ca_s42/best.pth \
      --dataset haze4k --root /data/Haze4K/test --seed 42
  python eval_v2.py --model sa_ca_nohier --ckpt runs/nohier_s42/best.pth \
      --dataset haze4k --root /data/Haze4K/test --seed 42
  # 跨域:
  python eval_v2.py --model sa_ca --ckpt ... --dataset nhhaze --root /data/NH-HAZE
"""
import os, argparse, time, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from build_any import build_any as build_model

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def set_seed(s=42):
    os.environ['PYTHONHASHSEED'] = str(s); random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def psnr(pred, tgt, eps=1e-10):
    mse = torch.mean((pred - tgt) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse))


def _gauss(ws, sigma, dev):
    c = torch.arange(ws, device=dev).float() - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); return g / g.sum()


def ssim(pred, tgt, ws=11, sigma=1.5):
    dev = pred.device; c1, c2 = 0.01 ** 2, 0.03 ** 2
    g = _gauss(ws, sigma, dev); w1 = g.view(1, 1, 1, -1); w2 = g.view(1, 1, -1, 1)
    def filt(x):
        x = F.conv2d(x, w1.expand(x.size(1), 1, 1, ws), padding=(0, ws // 2), groups=x.size(1))
        x = F.conv2d(x, w2.expand(x.size(1), 1, ws, 1), padding=(ws // 2, 0), groups=x.size(1))
        return x
    mx, my = filt(pred), filt(tgt)
    sx = filt(pred * pred) - mx * mx; sy = filt(tgt * tgt) - my * my
    sxy = filt(pred * tgt) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))
    return float(m.mean())


def read01(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def pairs_haze4k(root):
    root = Path(root); ir, tr = root / 'haze', root / 'gt'; out = []
    for p in sorted(ir.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            i = p.stem.split('_')[0]; g = None
            for e in ['.png', '.jpg', '.jpeg', '.bmp']:
                if (tr / (i + e)).exists(): g = tr / (i + e); break
            if g is None and (tr / p.name).exists(): g = tr / p.name
            if g and g.exists(): out.append((p, g))
    return out


def pairs_nhhaze(root):
    root = Path(root); out = []
    # 结构1: 划分后的 haze/ + gt/ 子目录(同名配对), split_nhhaze.py 产物 / 与训练一致
    hz_dir, gt_dir = root / 'haze', root / 'gt'
    if hz_dir.is_dir() and gt_dir.is_dir():
        for p in sorted(hz_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                g = gt_dir / p.name
                if not g.exists():  # 兼容 id 配对 (X_*.png -> X.png)
                    i = p.stem.split('_')[0]
                    for e in ('.png', '.jpg', '.jpeg', '.bmp'):
                        if (gt_dir / (i + e)).exists():
                            g = gt_dir / (i + e); break
                if g.exists(): out.append((p, g))
        if out: return out
    # 结构2: 原始 *_hazy / *_GT 同目录命名
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and '_hazy' in p.stem.lower():
            g = p.parent / p.name.replace('_hazy', '_GT').replace('_Hazy', '_GT')
            if g.exists(): out.append((p, g))
    return out


@torch.no_grad()
def infer_full(model, x):
    o = model(x)
    return (o[0] if isinstance(o, (list, tuple)) else o)


@torch.no_grad()
def infer_tile(model, x, tile=64, overlap=32):
    b, c, h, w = x.shape; stride = tile - overlap
    out = torch.zeros((1, 3, h, w), device=x.device); wt = torch.zeros((1, 1, h, w), device=x.device)
    wy = torch.hann_window(tile, device=x.device).view(1, 1, tile, 1)
    wx = torch.hann_window(tile, device=x.device).view(1, 1, 1, tile)
    w2 = (wy * wx).clamp_min(1e-6)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = x[:, :, y0:y1, x0:x1]; ph, pw = y1 - y0, x1 - x0
            if ph < tile or pw < tile:
                patch = F.pad(patch, (0, tile - pw, 0, tile - ph), mode='reflect')
            pred = infer_full(model, patch)[:, :, :ph, :pw]
            out[:, :, y0:y1, x0:x1] += pred * w2[:, :, :ph, :pw]
            wt[:, :, y0:y1, x0:x1] += w2[:, :, :ph, :pw]
    return out / wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sa_ca')
    ap.add_argument('--channel', type=int, default=32)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', choices=['haze4k', 'nhhaze'], required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--tile', type=int, default=0, help='>0 则滑窗推理(对照用); 0=整图')
    ap.add_argument('--overlap', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(args.model, in_channel=3, channel=args.channel).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True); model.eval()

    pairs = pairs_haze4k(args.root) if args.dataset == 'haze4k' else pairs_nhhaze(args.root)
    print(f'{len(pairs)} pairs | model={args.model} | tile={args.tile or "full"} | dev={dev}')
    if not pairs:
        print('ERROR: no pairs'); return
    if args.save_dir: os.makedirs(args.save_dir, exist_ok=True)

    ps, ss, ms = [], [], []
    for i, (ip, gp) in enumerate(pairs, 1):
        x = read01(ip).to(dev); gt = read01(gp).to(dev)
        if x.shape != gt.shape:
            gt = F.interpolate(gt, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if dev == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = (infer_tile(model, x, args.tile, args.overlap) if args.tile > 0
                else infer_full(model, x)).clamp(0, 1)
        if dev == 'cuda': torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        ps.append(psnr(pred, gt)); ss.append(ssim(pred, gt))
        if args.save_dir:
            from torchvision.utils import save_image
            save_image(pred[0], os.path.join(args.save_dir, ip.name))
        if i % 50 == 0 or i == len(pairs):
            print(f'  {i}/{len(pairs)} avgPSNR={np.mean(ps):.2f} avgSSIM={np.mean(ss):.4f}')

    res = {'model': args.model, 'dataset': args.dataset, 'seed': args.seed,
           'tile': args.tile, 'count': len(pairs),
           'psnr': float(np.mean(ps)), 'ssim': float(np.mean(ss)),
           'ms_per_image': float(np.mean(ms)),
           'fps': float(1000.0 / np.mean(ms))}
    print(f"\nPSNR={res['psnr']:.4f} SSIM={res['ssim']:.4f} "
          f"FPS={res['fps']:.2f} ({res['ms_per_image']:.1f}ms)")
    fn = f"result_{args.model}_{args.dataset}_seed{args.seed}.json"
    json.dump(res, open(fn, 'w'), indent=2)
    print(f'Saved -> {fn}')


if __name__ == '__main__':
    main()
