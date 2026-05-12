import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import FullAudioDataset, LibriSpeechTrainDataset, collate_audio
from .losses import MultiScaleMelSpectrogramLoss


def pad(k, d=1):
    return (k - 1) * d // 2


class Snake(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, c, 1))

    def forward(self, x):
        return x + torch.sin(self.a * x).pow(2) / (self.a + 1e-9)


class Res(nn.Module):
    def __init__(self, c, d, k=7):
        super().__init__()
        h = c // 2
        self.net = nn.Sequential(Snake(c), nn.Conv1d(c, h, k, padding=pad(k, d), dilation=d), Snake(h), nn.Conv1d(h, c, 1))

    def forward(self, x):
        return x + self.net(x)


class EncBlock(nn.Module):
    def __init__(self, c, s, ds, k):
        super().__init__()
        self.net = nn.Sequential(
            Res(c // 2, ds[0], k),
            Res(c // 2, ds[1], k),
            Res(c // 2, ds[2], k),
            Snake(c // 2),
            nn.Conv1d(c // 2, c, 2 * s, stride=s, padding=math.ceil(s / 2)),
        )

    def forward(self, x):
        return self.net(x)


class DecBlock(nn.Module):
    def __init__(self, c, s, ds, k):
        super().__init__()
        self.net = nn.Sequential(
            Snake(c),
            nn.ConvTranspose1d(c, c // 2, 2 * s, stride=s, padding=math.ceil(s / 2), output_padding=2 * math.ceil(s / 2) - s),
            Res(c // 2, ds[0], k),
            Res(c // 2, ds[1], k),
            Res(c // 2, ds[2], k),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, base, z, strides, ds, k):
        super().__init__()
        layers = [nn.Conv1d(1, base, 7, padding=3)]
        c = base
        for s in strides:
            layers.append(EncBlock(c * 2, s, ds, k))
            c *= 2
        layers += [Snake(c), nn.Conv1d(c, z, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, base, z, strides, ds, k):
        super().__init__()
        c = base * (2 ** len(strides))
        layers = [nn.Conv1d(z, c, 7, padding=3)]
        for s in reversed(strides):
            layers.append(DecBlock(c, s, ds, k))
            c //= 2
        layers += [Snake(base), nn.Conv1d(base, 1, 7, padding=3), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QOut:
    def __init__(self, q, commit, code, ppl, ids):
        self.quantized = q
        self.commitment_loss = commit
        self.codebook_loss = code
        self.perplexity = ppl
        self.codes = ids


class VQ(nn.Module):
    def __init__(self, dim, n, decay, eps, dead):
        super().__init__()
        emb = torch.randn(n, dim)
        self.dim = dim
        self.n = n
        self.decay = decay
        self.eps = eps
        self.dead = dead
        self.register_buffer("emb", emb)
        self.register_buffer("cnt", torch.zeros(n))
        self.register_buffer("avg", emb.clone())
        self.ready = False

    @torch.no_grad()
    def init(self, x):
        if self.ready:
            return
        c = x[torch.randperm(x.size(0), device=x.device)[: self.n]].clone()
        for _ in range(10):
            d = x.pow(2).sum(1, keepdim=True) - 2 * x @ c.t() + c.pow(2).sum(1)
            ids = d.argmin(1)
            for i in range(self.n):
                m = ids == i
                if m.any():
                    c[i] = x[m].mean(0)
        self.emb.copy_(c)
        self.avg.copy_(c)
        self.ready = True

    def forward(self, x):
        dtype = x.dtype
        flat = x.permute(0, 2, 1).reshape(-1, self.dim).to(self.emb.dtype)
        if self.training and not self.ready and flat.size(0) >= self.n:
            self.init(flat)

        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.emb.t() + self.emb.pow(2).sum(1)
        ids = d.argmin(1)
        onehot = F.one_hot(ids, self.n).type_as(flat)
        q = F.embedding(ids, self.emb).view(x.size(0), x.size(2), x.size(1)).permute(0, 2, 1).to(dtype)

        if self.training:
            with torch.no_grad():
                cnt = onehot.detach().sum(0)
                avg = onehot.detach().t() @ flat.detach()
                self.cnt.mul_(self.decay).add_(cnt, alpha=1 - self.decay)
                self.avg.mul_(self.decay).add_(avg, alpha=1 - self.decay)
                cnt = (self.cnt + self.eps) / (self.cnt.sum() + self.n * self.eps) * self.cnt.sum()
                self.emb.copy_(self.avg / cnt.unsqueeze(1))
                dead = self.cnt < self.dead
                if dead.any():
                    r = torch.randint(0, flat.size(0), (int(dead.sum().item()),), device=flat.device)
                    self.emb[dead] = flat.detach()[r].to(self.emb.dtype)
                    self.avg[dead] = flat.detach()[r].to(self.emb.dtype)
                    self.cnt[dead] = self.dead

        p = onehot.float().mean(0)
        ppl = torch.exp(-(p * (p + 1e-10).log()).sum())
        return q, ppl, ids.view(x.size(0), x.size(2))


class RVQ(nn.Module):
    def __init__(self, dim, n, levels, decay, eps, dead):
        super().__init__()
        self.qs = nn.ModuleList([VQ(dim, n, decay, eps, dead) for _ in range(levels)])

    def forward(self, x):
        res = x
        qsum = torch.zeros_like(x)
        ppls = []
        ids = []
        for q in self.qs:
            z, ppl, code = q(res)
            qsum = qsum + z
            res = res - z
            ppls.append(ppl)
            ids.append(code)
        return QOut(x + (qsum - x).detach(), F.mse_loss(x, qsum.detach()), F.mse_loss(qsum, x.detach()), torch.stack(ppls).mean(), torch.stack(ids, dim=1))


class SoundStream(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.encoder = Encoder(m["base_channels"], m["latent_dim"], m["strides"], m["dilations"], m["residual_kernel_size"])
        self.quantizer = RVQ(m["latent_dim"], m["codebook_size"], m["num_quantizers"], m["quantizer_decay"], m["quantizer_eps"], m["quantizer_replace_threshold"])
        self.decoder = Decoder(m["base_channels"], m["latent_dim"], m["strides"], m["dilations"], m["residual_kernel_size"])

    def forward(self, x):
        z = self.encoder(x)
        q = self.quantizer(z)
        y = self.decoder(q.quantized)
        return {"audio": y, "commitment_loss": q.commitment_loss, "codebook_loss": q.codebook_loss, "perplexity": q.perplexity, "codes": q.codes}


def act():
    return nn.LeakyReLU(0.2)


class D1(nn.Module):
    def __init__(self, c, sn=False):
        super().__init__()
        norm = nn.utils.spectral_norm if sn else nn.utils.weight_norm
        ch = [1, c, c * 4, c * 16, c * 64, c * 64, c * 64]
        ks = [15, 41, 41, 41, 41, 5]
        st = [1, 4, 4, 4, 4, 1]
        gs = [1, 4, 16, 64, 256, 1]
        self.blocks = nn.ModuleList([norm(nn.Conv1d(ch[i], ch[i + 1], ks[i], stride=st[i], padding=(ks[i] - 1) // 2, groups=gs[i])) for i in range(len(ks))])
        self.last = norm(nn.Conv1d(ch[-1], 1, 3, padding=1))

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = act()(b(x))
            feats.append(x)
        return self.last(x), feats


class WaveD(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ds = nn.ModuleList([D1(c, True), D1(c), D1(c)])
        self.pool = nn.AvgPool1d(4, stride=2, padding=2)

    def forward(self, x):
        out = []
        for i, d in enumerate(self.ds):
            if i > 0:
                x = self.pool(x)
            out.append(d(x))
        return out


class STFTD1(nn.Module):
    def __init__(self, c):
        super().__init__()
        ch = [2, c, c, c * 2, c * 4, c * 4]
        st = [(1, 2), (2, 2), (1, 2), (2, 2), (1, 2)]
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, stride=st[i], padding=1), act()) for i in range(len(st))])
        self.last = nn.Conv2d(ch[-1], 1, 3, padding=1)

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        return self.last(x), feats


class STFTD(nn.Module):
    def __init__(self, c, ffts, hops, wins):
        super().__init__()
        self.ffts = ffts
        self.hops = hops
        self.wins = wins
        self.ds = nn.ModuleList([STFTD1(c) for _ in ffts])

    def make(self, x, n, h, w):
        win = torch.hann_window(w, device=x.device)
        s = torch.stft(x.squeeze(1), n_fft=n, hop_length=h, win_length=w, window=win, return_complex=True)
        return torch.stack([s.real, s.imag], dim=1)

    def forward(self, x):
        out = []
        for d, n, h, w in zip(self.ds, self.ffts, self.hops, self.wins):
            out.append(d(self.make(x, n, h, w)))
        return out


class Disc(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["discriminator"]
        self.wave = WaveD(d["wave_channels"])
        self.stft = STFTD(d["stft_channels"], d["stft_fft_sizes"], d["stft_hop_sizes"], d["stft_win_lengths"])

    def forward(self, x):
        return {"wave": self.wave(x), "stft": self.stft(x)}


def build_model(cfg):
    return SoundStream(cfg)


def build_discriminator(cfg):
    return Disc(cfg)


def build_mel_loss(cfg):
    a = cfg["audio"]
    return MultiScaleMelSpectrogramLoss(a["sample_rate"], a["n_mels"], a["mel_fft_sizes"], a["mel_hop_sizes"], a["mel_win_lengths"], a["mel_fmin"], a["mel_fmax"])


def build_train_loader(cfg):
    ds = LibriSpeechTrainDataset(cfg["data"]["train_root"], cfg["audio"]["sample_rate"], cfg["audio"]["crop_seconds"])
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True, collate_fn=collate_audio)


def build_eval_loader(cfg):
    ds = FullAudioDataset(cfg["data"]["valid_root"], cfg["audio"]["sample_rate"], cfg["data"].get("eval_num_files"))
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
