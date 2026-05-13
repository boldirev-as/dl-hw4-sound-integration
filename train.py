import itertools
import random
import sys
from pathlib import Path

import comet_ml
import numpy as np
import torch
import torchaudio
import yaml
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from soundstream.losses import discriminator_loss, feature_matching_loss, generator_adversarial_loss
from soundstream.metrics import MetricTracker, average_metrics
from soundstream.model import build_discriminator, build_eval_loader, build_mel_loss, build_model, build_train_loader


def flat(x):
    return x["wave"] + x["stft"]


def save(path, net, disc, opt_g, opt_d, i, cfg):
    torch.save(
        {
            "model": net.state_dict(),
            "discriminator": disc.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "step": i,
            "config": cfg,
        },
        path,
    )


@torch.no_grad()
def val(net, mel, dl, dev, sr):
    net.eval()
    mt = MetricTracker(sr, True)
    ms = []
    xs = []
    for wav, _ in dl:
        wav = wav.to(dev)
        rec = net(wav)["audio"][..., : wav.size(-1)]
        rec = torch.nan_to_num(rec, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        ms.append(float(mel(rec, wav).item()))
        xs.append(mt.update(rec, wav))
    out = average_metrics(xs)
    out["mel"] = sum(ms) / len(ms)
    net.train()
    return out


def log(exp, data, i):
    for k, v in data.items():
        exp.log_metric(k, v, step=i)


def one_step(net, disc, mel, opt_g, opt_d, sc, wav, dev, cfg):
    wav = wav.to(dev, non_blocking=True).clamp(-1, 1)
    tr = cfg["train"]
    lw = cfg["losses"]
    amp = tr["amp"] and dev.type == "cuda"

    with autocast("cuda", enabled=amp):
        out = net(wav)
        rec = out["audio"][..., : wav.size(-1)].clamp(-1, 1)
        fake = flat(disc(rec.detach()))
        real = flat(disc(wav))
        d_loss = discriminator_loss(real, fake)

    opt_d.zero_grad(set_to_none=True)
    if torch.isfinite(d_loss):
        sc.scale(d_loss).backward()
        sc.unscale_(opt_d)
        nn.utils.clip_grad_norm_(disc.parameters(), tr["clip_grad_norm"])
        sc.step(opt_d)

    with autocast("cuda", enabled=amp):
        out = net(wav)
        rec = out["audio"][..., : wav.size(-1)].clamp(-1, 1)
        fake = flat(disc(rec))
        real = flat(disc(wav))
        mel_loss = mel(rec, wav)
        adv = generator_adversarial_loss(fake)
        fm = feature_matching_loss(real, fake)
        commit = out["commitment_loss"]
        code = out["codebook_loss"]
        g_loss = (
            lw["mel_weight"] * mel_loss
            + lw["feature_matching_weight"] * fm
            + lw["adversarial_weight"] * adv
            + lw["commitment_weight"] * commit
            + code
        )

    opt_g.zero_grad(set_to_none=True)
    if torch.isfinite(g_loss):
        sc.scale(g_loss).backward()
        sc.unscale_(opt_g)
        nn.utils.clip_grad_norm_(net.parameters(), tr["clip_grad_norm"])
        sc.step(opt_g)
    sc.update()

    return {
        "g": float(g_loss.item()),
        "d": float(d_loss.item()),
        "mel": float(mel_loss.item()),
        "adv": float(adv.item()),
        "fm": float(fm.item()),
        "commit": float(commit.item()),
        "code": float(code.item()),
        "ppl": float(out["perplexity"].item()),
    }


def main():
    with Path("configs/config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tr = cfg["train"]

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    out_dir = Path(tr["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    wav_dir = out_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    net = build_model(cfg).to(dev)
    disc = build_discriminator(cfg).to(dev)
    mel = build_mel_loss(cfg).to(dev)

    opt_g = torch.optim.AdamW(net.parameters(), lr=tr["learning_rate"], betas=tuple(tr["betas"]))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=tr["learning_rate"], betas=tuple(tr["betas"]))
    sc = GradScaler("cuda", enabled=tr["amp"] and dev.type == "cuda")

    start = 0
    if tr["resume"]:
        ckpt = torch.load(tr["resume"], map_location=dev)
        net.load_state_dict(ckpt["model"])
        disc.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["optimizer_g"])
        opt_d.load_state_dict(ckpt["optimizer_d"])
        start = int(ckpt["step"])

    train_dl = build_train_loader(cfg)
    val_dl = build_eval_loader(cfg)
    data = itertools.cycle(train_dl)

    exp = comet_ml.Experiment(
        project_name=cfg["logging"]["comet_project_name"],
        workspace=cfg["logging"]["comet_workspace"],
    )
    exp.set_name(cfg["logging"]["comet_experiment_name"])
    exp.add_tags(cfg["logging"]["comet_tags"])
    exp.log_parameters(cfg)

    bar = tqdm(range(start, tr["max_steps"]), initial=start, total=tr["max_steps"])
    for i in bar:
        m = one_step(net, disc, mel, opt_g, opt_d, sc, next(data), dev, cfg)
        bar.set_postfix(g=f"{m['g']:.3f}", d=f"{m['d']:.3f}", mel=f"{m['mel']:.3f}", ppl=f"{m['ppl']:.2f}")

        if i % tr["log_interval"] == 0:
            log(exp, m, i)

        if i > 0 and i % tr["eval_interval"] == 0:
            vm = val(net, mel, val_dl, dev, cfg["audio"]["sample_rate"])
            log(exp, {f"eval_{k}": v for k, v in vm.items()}, i)

            wav, name = next(iter(val_dl))
            wav = wav.to(dev)
            with torch.no_grad():
                rec = net(wav)["audio"][..., : wav.size(-1)].cpu()
            path = wav_dir / f"{i:07d}_{name[0]}.wav"
            torchaudio.save(path, rec[0], cfg["audio"]["sample_rate"])
            exp.log_audio(str(path), file_name=path.name, sample_rate=cfg["audio"]["sample_rate"], step=i)

        if i > 0 and i % tr["save_interval"] == 0:
            save(ckpt_dir / f"step_{i:07d}.pt", net, disc, opt_g, opt_d, i, cfg)

    save(ckpt_dir / "final.pt", net, disc, opt_g, opt_d, tr["max_steps"], cfg)
    exp.end()


if __name__ == "__main__":
    main()
