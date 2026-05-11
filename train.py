import itertools
import sys
from pathlib import Path

import torch
import torchaudio
from comet_ml import Experiment
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from soundstream.config import load_config
from soundstream.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from soundstream.metrics import MetricTracker, average_metrics
from soundstream.model import build_discriminator, build_eval_loader, build_mel_loss, build_model, build_train_loader
from soundstream.utils import count_parameters, ensure_dir, seed_everything


def flatten_disc_outputs(outputs):
    return outputs["wave"] + outputs["stft"]


def save_checkpoint(path, model, discriminator, optimizer_g, optimizer_d, step, config):
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model,
    mel_loss_fn,
    eval_loader,
    device,
    sample_rate,
    use_nisqa,
):
    model.eval()
    tracker = MetricTracker(sample_rate=sample_rate, use_nisqa=use_nisqa)
    mel_values = []
    metric_values = []
    for audio, _ in eval_loader:
        audio = audio.to(device)
        output = model(audio)
        reconstructed = output["audio"][..., : audio.size(-1)]
        mel_values.append(float(mel_loss_fn(reconstructed, audio).item()))
        metric_values.append(tracker.update(reconstructed, audio))
    metrics = average_metrics(metric_values)
    metrics["mel"] = sum(mel_values) / len(mel_values)
    model.train()
    return metrics


def log_comet(experiment, data, step):
    for key, value in data.items():
        experiment.log_metric(key, value, step=step)


def maybe_init_comet(config):
    logging_config = config["logging"]
    experiment = Experiment(
        project_name=logging_config["comet_project_name"],
        workspace=logging_config["comet_workspace"],
    )
    experiment.set_name(logging_config["comet_experiment_name"])
    experiment.add_tags(logging_config["comet_tags"])
    experiment.log_parameters(config)
    return experiment


def train_step(
    model,
    discriminator,
    mel_loss_fn,
    optimizer_g,
    optimizer_d,
    scaler,
    batch,
    device,
    config,
):
    batch = batch.to(device, non_blocking=True)
    batch = batch.clamp(-1.0, 1.0)

    losses_config = config["losses"]
    train_config = config["train"]
    amp_enabled = train_config["amp"] and device.type == "cuda"

    with autocast(enabled=amp_enabled):
        model_output = model(batch)
        reconstructed = model_output["audio"][..., : batch.size(-1)]
        fake_disc_outputs = flatten_disc_outputs(discriminator(reconstructed.detach()))
        real_disc_outputs = flatten_disc_outputs(discriminator(batch))
        d_loss = discriminator_loss(real_disc_outputs, fake_disc_outputs)

    optimizer_d.zero_grad(set_to_none=True)
    scaler.scale(d_loss).backward()
    scaler.unscale_(optimizer_d)
    nn.utils.clip_grad_norm_(discriminator.parameters(), train_config["clip_grad_norm"])
    scaler.step(optimizer_d)

    with autocast(enabled=amp_enabled):
        model_output = model(batch)
        reconstructed = model_output["audio"][..., : batch.size(-1)]
        fake_disc_outputs = flatten_disc_outputs(discriminator(reconstructed))
        real_disc_outputs = flatten_disc_outputs(discriminator(batch))
        mel_loss = mel_loss_fn(reconstructed, batch)
        adv_loss = generator_adversarial_loss(fake_disc_outputs)
        fm_loss = feature_matching_loss(real_disc_outputs, fake_disc_outputs)
        commitment_loss = model_output["commitment_loss"]
        codebook_loss = model_output["codebook_loss"]
        g_loss = (
            losses_config["mel_weight"] * mel_loss
            + losses_config["feature_matching_weight"] * fm_loss
            + losses_config["adversarial_weight"] * adv_loss
            + losses_config["commitment_weight"] * commitment_loss
            + codebook_loss
        )

    optimizer_g.zero_grad(set_to_none=True)
    scaler.scale(g_loss).backward()
    scaler.unscale_(optimizer_g)
    nn.utils.clip_grad_norm_(model.parameters(), train_config["clip_grad_norm"])
    scaler.step(optimizer_g)
    scaler.update()

    return {
        "generator_loss": float(g_loss.item()),
        "discriminator_loss": float(d_loss.item()),
        "mel_loss": float(mel_loss.item()),
        "adv_loss": float(adv_loss.item()),
        "feature_matching_loss": float(fm_loss.item()),
        "commitment_loss": float(commitment_loss.item()),
        "codebook_loss": float(codebook_loss.item()),
        "perplexity": float(model_output["perplexity"].item()),
    }


def main():
    config = load_config("configs/config.yaml")

    seed_everything(config["seed"])
    output_dir = ensure_dir(config["train"]["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    samples_dir = ensure_dir(output_dir / "samples")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    discriminator = build_discriminator(config).to(device)
    mel_loss_fn = build_mel_loss(config).to(device)

    optimizer_g = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=tuple(config["train"]["betas"]),
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config["train"]["learning_rate"],
        betas=tuple(config["train"]["betas"]),
    )
    scaler = GradScaler(enabled=config["train"]["amp"] and device.type == "cuda")

    start_step = 0
    if config["train"]["resume"]:
        checkpoint = torch.load(config["train"]["resume"], map_location=device)
        model.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_step = int(checkpoint["step"])

    train_loader = build_train_loader(config)
    eval_loader = build_eval_loader(config)

    experiment = maybe_init_comet(config)
    log_comet(
        experiment,
        {
            "num_parameters_generator": count_parameters(model),
            "num_parameters_discriminator": count_parameters(discriminator),
        },
        step=start_step,
    )

    data_iterator = itertools.cycle(train_loader)
    progress = tqdm(range(start_step, config["train"]["max_steps"]), initial=start_step, total=config["train"]["max_steps"])
    for step in progress:
        batch = next(data_iterator)
        train_metrics = train_step(
            model=model,
            discriminator=discriminator,
            mel_loss_fn=mel_loss_fn,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scaler=scaler,
            batch=batch,
            device=device,
            config=config,
        )

        progress.set_postfix(
            generator=f"{train_metrics['generator_loss']:.3f}",
            discriminator=f"{train_metrics['discriminator_loss']:.3f}",
            mel=f"{train_metrics['mel_loss']:.3f}",
            perplexity=f"{train_metrics['perplexity']:.2f}",
        )

        if step % config["train"]["log_interval"] == 0:
            log_comet(experiment, train_metrics, step=step)

        if step > 0 and step % config["train"]["eval_interval"] == 0:
            eval_metrics = evaluate(
                model=model,
                mel_loss_fn=mel_loss_fn,
                eval_loader=eval_loader,
                device=device,
                sample_rate=config["audio"]["sample_rate"],
                use_nisqa=True,
            )
            log_comet(experiment, {f"eval/{key}": value for key, value in eval_metrics.items()}, step=step)

            sample_audio, sample_name = next(iter(eval_loader))
            sample_audio = sample_audio.to(device)
            with torch.no_grad():
                sample_output = model(sample_audio)
            reconstructed = sample_output["audio"][..., : sample_audio.size(-1)].cpu()
            sample_path = samples_dir / f"{step:07d}_{sample_name[0]}_reconstructed.wav"
            torchaudio.save(sample_path, reconstructed[0], config["audio"]["sample_rate"])
            if experiment is not None:
                experiment.log_audio(
                    file_data=str(sample_path),
                    file_name=sample_path.name,
                    sample_rate=config["audio"]["sample_rate"],
                    step=step,
                    metadata={"source_file": sample_name[0]},
                )

        if step > 0 and step % config["train"]["save_interval"] == 0:
            save_checkpoint(
                checkpoints_dir / f"step_{step:07d}.pt",
                model=model,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                step=step,
                config=config,
            )

    save_checkpoint(
        checkpoints_dir / "final.pt",
        model=model,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        step=config["train"]["max_steps"],
        config=config,
    )
    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
