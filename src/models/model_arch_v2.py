"""
model_arch_v2.py — 分辨率灵活版 + 核心消融变体（P1/P2 实验用）
================================================================
相对原 model_arch.py 的改动（全部为"加法"，不破坏原文已发表结构）：

1. 解除 `assert x.shape==64`：forward 内自动 reflect-pad 到合法倍数，
   推理结束裁回原尺寸 → 支持 256/512/1024 等任意整图推理（编辑 #5/#7）。
2. 参数化：window_size / n_stages(下采样次数) / hier(是否分层) / variant。
3. 内置核心消融变体 hier=False：保持 **完全相同的深度、通道调度、block 类型、
   多尺度监督**，仅把"下采样"替换为 stride=1（空间不变）、"上采样"替换为 identity，
   使瓶颈窗口永远只覆盖 8x8 局部 → 直接验证"分层下采样=全局感受野"这一核心 claim。
4. forward 返回 list（长度 = n_stages+1），由 train_v2.py 做"自适应多尺度 loss"。

设计原则：除被消融的那一个变量外，其余一切保持与正文模型一致，保证对照公平。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


# ==================== 注意力模块（与正文一致，window_size 改为可配） ====================

class Channel_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=False), nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Space_Attention(nn.Module):
    def __init__(self, dim, head_num=4, window_size=8):
        super().__init__()
        assert (dim % head_num) == 0
        self.head_dim = dim // head_num
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.to_qkv = nn.Sequential(
            nn.Conv2d(dim, 3 * dim, 1, 1, 0), nn.ReLU(True),
            nn.Conv2d(3 * dim, 3 * dim, 3, 1, 1, groups=3 * dim))
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        ws = self.window_size
        # 防御：若某层分辨率 < window 或不整除，临时 pad 到整除（推理整图时已在外层保证整除）
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H, W = x.shape[2], x.shape[3]
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (hd d) (x w1) (y w2) -> (b x y) hd (w1 w2) d',
                                d=self.head_dim, w1=ws, w2=ws), (q, k, v))
        q = q * self.scale
        sim = einsum(q, k, 'b hd i d, b hd j d -> b hd i j')
        attn = F.softmax(sim, dim=-1)
        out = einsum(attn, v, 'b hd i j, b hd j d -> b hd i d')
        out = rearrange(out, '(b x y) hd (w1 w2) d -> b (hd d) (x w1) (y w2)',
                        x=H // ws, y=W // ws, w1=ws, w2=ws)
        out = self.to_out(out)
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


def _make_block(dim, variant, head_num, window_size):
    return _TFBlock(dim, variant, head_num, window_size)


class _TFBlock(nn.Module):
    """统一 block：variant ∈ {'sa_ca','sa','ca'}"""
    def __init__(self, dim, variant='sa_ca', head_num=4, window_size=8):
        super().__init__()
        self.variant = variant
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.use_sa = 'sa' in variant
        self.use_ca = 'ca' in variant
        if self.use_sa:
            self.sa = Space_Attention(dim, head_num=head_num, window_size=window_size)
        if self.use_ca:
            self.ca = Channel_Attention(dim)
        if self.use_sa and self.use_ca:
            self.fuse = nn.Conv2d(dim, dim, 1, 1, 0)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0), nn.GELU(), nn.Conv2d(dim, dim, 1, 1, 0))

    def forward(self, x):
        inp = x
        y = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.use_sa and self.use_ca:
            y = self.fuse(self.sa(y) + self.ca(y))
        elif self.use_sa:
            y = self.sa(y)
        else:
            y = self.ca(y)
        x = y + inp
        inp = x
        y = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ffn(y) + inp
        return x


# ==================== 主模型 ====================

class UTNetV2(nn.Module):
    """
    分辨率灵活的 U 形 Transformer + 可消融。

    参数:
      variant      : 'sa_ca' | 'sa' | 'ca'      (注意力分支消融, 编辑 #4)
      channel      : 基础通道 (默认 32)
      n_stages     : 下采样次数 (默认 3 → 64/32/16/8 四级; 可设 2/4 做 stage 消融)
      window_size  : 窗口大小 (默认 8; 可设 4/16 做窗口消融)
      head_num     : 注意力头数 (默认 4; 可做头数消融)
      hier         : True=分层下采样(正文); False=无分层(核心消融, 空间恒定)
    """
    def __init__(self, in_channel=3, channel=32, variant='sa_ca',
                 n_stages=3, window_size=8, head_num=4, hier=True):
        super().__init__()
        self.in_channel = in_channel
        self.n_stages = n_stages
        self.window_size = window_size
        self.hier = hier
        # 整图推理时输入需 pad 到的倍数:
        #   分层: window_size * 2**n_stages (保证瓶颈层整除窗口)
        #   无分层: window_size
        self.size_mult = window_size * (2 ** n_stages) if hier else window_size

        B = lambda d: _make_block(d, variant, head_num, window_size)

        self.conv_in = nn.Conv2d(in_channel, channel, 3, 1, 1)

        # 编码器 + 下采样
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        dims = [channel]
        c = channel
        for _ in range(n_stages):
            self.encoders.append(B(c))
            nc = 2 * c
            if hier:
                self.downs.append(nn.Conv2d(c, nc, 3, 2, 1))       # 空间/2
            else:
                self.downs.append(nn.Conv2d(c, nc, 3, 1, 1))       # 空间不变(消融)
            c = nc
            dims.append(c)

        # 瓶颈
        self.bottleneck = B(c)
        self.head_bottleneck = nn.Conv2d(c, in_channel, 3, 1, 1)

        # 解码器 + 上采样 + 各级输出头
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList()
        for s in range(n_stages):
            skip_c = dims[n_stages - 1 - s]   # 对应编码器 skip 的通道
            if hier:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(c, skip_c, 1, 1, 0)))
            else:
                self.ups.append(nn.Conv2d(c, skip_c, 1, 1, 0))     # 仅改通道(消融)
            self.decoders.append(B(skip_c))
            self.heads.append(nn.Conv2d(skip_c, in_channel, 3, 1, 1))
            c = skip_c

    def _pad_to_mult(self, x):
        _, _, h, w = x.shape
        m = self.size_mult
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, h, w

    def forward(self, x):
        """返回多尺度输出 list: [finest(原图分辨率), ..., coarsest(瓶颈分辨率)]
        训练时全部参与自适应 loss; 推理只取 outs[0]。"""
        x, h0, w0 = self._pad_to_mult(x)
        x = self.conv_in(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            xe = enc(x)
            skips.append(xe)
            x = down(xe)

        xb = self.bottleneck(x)
        outs_coarse_to_fine = [self.head_bottleneck(xb)]

        x = xb
        for up, dec, head, skip in zip(self.ups, self.decoders, self.heads,
                                       reversed(skips)):
            x = up(x) + skip
            x = dec(x)
            outs_coarse_to_fine.append(head(x))

        # 反转为 finest->coarsest, 并把最细输出裁回原始尺寸
        outs = outs_coarse_to_fine[::-1]
        outs[0] = outs[0][:, :, :h0, :w0]
        return outs


# ==================== 注册表 ====================

def build_model(name, in_channel=3, channel=32):
    """便捷构造: name 形如 'sa_ca' / 'sa' / 'ca' / 'sa_ca_nohier' / 'sa_ca_ws4' ..."""
    cfg = dict(variant='sa_ca', channel=channel, n_stages=3,
               window_size=8, head_num=4, hier=True, in_channel=in_channel)
    if name.startswith('sa_ca'):
        cfg['variant'] = 'sa_ca'
    elif name.startswith('sa'):
        cfg['variant'] = 'sa'
    elif name.startswith('ca'):
        cfg['variant'] = 'ca'
    if 'nohier' in name:
        cfg['hier'] = False
    for tok in name.split('_'):
        if tok.startswith('ws'):
            cfg['window_size'] = int(tok[2:])
        if tok.startswith('st') and tok[2:].isdigit():
            cfg['n_stages'] = int(tok[2:])
        if tok.startswith('hd') and tok[2:].isdigit():
            cfg['head_num'] = int(tok[2:])
    return UTNetV2(**cfg)


MODEL_REGISTRY_V2 = {
    'sa_ca':        lambda **kw: UTNetV2(variant='sa_ca', **kw),
    'sa':           lambda **kw: UTNetV2(variant='sa', **kw),
    'ca':           lambda **kw: UTNetV2(variant='ca', **kw),
    'sa_ca_nohier': lambda **kw: UTNetV2(variant='sa_ca', hier=False, **kw),  # 核心消融
}


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    for tag, kw in [('hier(正文)', dict(hier=True)),
                    ('no-hier(消融)', dict(hier=False))]:
        m = UTNetV2(channel=32, **kw).eval()
        x = torch.randn(1, 3, 64, 64)
        outs = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f'\n=== {tag} === params={n/1e6:.3f}M  outs={[tuple(o.shape) for o in outs]}')
        print(f'FLOPs@64: {FlopCountAnalysis(m, x).total()/1e9:.4f}G')
        # 验证任意分辨率
        for hw in [256, 512]:
            o = m(torch.randn(1, 3, hw, hw))[0]
            assert o.shape[-2:] == (hw, hw), o.shape
            print(f'  full-image {hw}x{hw} OK -> {tuple(o.shape)}')
