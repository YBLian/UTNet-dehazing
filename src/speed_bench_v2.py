"""
speed_bench_v2.py — 多分辨率 复杂度/速度 基准 (编辑 #5/#7)
==========================================================
对 UTNet(hier) / UTNet(no-hier) / 任意变体，在 64/256/512/1024 上测：
  Params, FLOPs(thop), 整图前向 FPS & latency(ms)。
全部为整图前向（非 64 滑窗）——这才是"全局感受野"卖点该被检验的方式。

用法:
  python speed_bench_v2.py --model sa_ca
  python speed_bench_v2.py --model sa_ca_nohier
  python speed_bench_v2.py --model sa_ca --sizes 64 512 1024 --repeat 200
结果写入 speed_<model>.json，可直接填论文表。
"""
import argparse, json, time, torch
from build_any import build_any as build_model


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_at(m, size, device):
    try:
        from thop import profile
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            f, _ = profile(m, inputs=(x,), verbose=False)
        return float(f)
    except Exception as e:
        print(f'  [FLOPs skip @ {size}] {e}')
        return None


@torch.no_grad()
def fps_at(m, size, device, warmup, repeat):
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        m(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        m(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeat
    return 1.0 / dt, dt * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sa_ca')
    ap.add_argument('--channel', type=int, default=32)
    ap.add_argument('--sizes', type=int, nargs='+', default=[64, 256, 512, 1024])
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--repeat', type=int, default=200)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = build_model(args.model, in_channel=3, channel=args.channel).to(device).eval()
    params = count_params(m)
    print(f'\nModel={args.model} channel={args.channel} device={device} '
          f'params={params:,} ({params/1e6:.3f}M)')

    rows = []
    for s in args.sizes:
        try:
            fl = flops_at(m, s, device)
            fps, ms = (None, None)
            if device == 'cuda':
                fps, ms = fps_at(m, s, device, args.warmup, args.repeat)
            else:
                # CPU 上也测但 repeat 调小，结果仅供参考
                fps, ms = fps_at(m, s, device, max(2, args.warmup // 10),
                                 max(5, args.repeat // 20))
            rows.append({'size': s, 'flops_g': (fl/1e9 if fl else None),
                         'fps': fps, 'ms': ms})
            print(f'  {s:>5}x{s:<5} | FLOPs={ (fl/1e9 if fl else float("nan")):8.3f}G '
                  f'| FPS={fps:8.2f} | {ms:8.2f} ms')
        except RuntimeError as e:
            print(f'  {s}: OOM/err -> {e}')
            rows.append({'size': s, 'error': str(e)})

    out = {'model': args.model, 'channel': args.channel,
           'params': params, 'device': device, 'rows': rows}
    fn = f'speed_{args.model}.json'
    json.dump(out, open(fn, 'w'), indent=2)
    print(f'\nSaved -> {fn}')


if __name__ == '__main__':
    main()
