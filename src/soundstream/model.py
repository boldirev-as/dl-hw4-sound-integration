import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import FullAudioDataset, LibriSpeechTrainDataset, collate_audio
from .discriminators import SoundStreamDiscriminator
from .losses import MultiScaleMelSpectrogramLoss
from .modules import SEANetDecoder, SEANetEncoder
from .quantizer import ResidualVectorQuantizer


class SoundStream(nn.Module):
    def __init__(
        self,
        sample_rate,
        base_channels,
        latent_dim,
        strides,
        residual_kernel_size,
        dilations,
        codebook_size,
        num_quantizers,
        quantizer_decay,
        quantizer_eps,
        quantizer_replace_threshold,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = SEANetEncoder(
            in_channels=1,
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides,
            dilations=dilations,
            residual_kernel_size=residual_kernel_size,
        )
        self.quantizer = ResidualVectorQuantizer(
            embedding_dim=latent_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            decay=quantizer_decay,
            eps=quantizer_eps,
            replace_threshold=quantizer_replace_threshold,
        )
        self.decoder = SEANetDecoder(
            out_channels=1,
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides,
            dilations=dilations,
            residual_kernel_size=residual_kernel_size,
        )

    def encode(self, audio):
        return self.encoder(audio)

    def decode(self, latents):
        return self.decoder(latents)

    def forward(self, audio):
        encoded = self.encode(audio)
        quantized = self.quantizer(encoded)
        reconstructed = self.decode(quantized.quantized)
        return {
            "audio": reconstructed,
            "encoded": encoded,
            "quantized": quantized.quantized,
            "commitment_loss": quantized.commitment_loss,
            "codebook_loss": quantized.codebook_loss,
            "perplexity": quantized.perplexity,
            "codes": quantized.codes,
        }


def build_model(config):
    model_config = config["model"]
    return SoundStream(
        sample_rate=config["audio"]["sample_rate"],
        base_channels=model_config["base_channels"],
        latent_dim=model_config["latent_dim"],
        strides=model_config["strides"],
        residual_kernel_size=model_config["residual_kernel_size"],
        dilations=model_config["dilations"],
        codebook_size=model_config["codebook_size"],
        num_quantizers=model_config["num_quantizers"],
        quantizer_decay=model_config["quantizer_decay"],
        quantizer_eps=model_config["quantizer_eps"],
        quantizer_replace_threshold=model_config["quantizer_replace_threshold"],
    )


def build_discriminator(config):
    disc_config = config["discriminator"]
    return SoundStreamDiscriminator(
        wave_channels=disc_config["wave_channels"],
        stft_channels=disc_config["stft_channels"],
        stft_fft_sizes=disc_config["stft_fft_sizes"],
        stft_hop_sizes=disc_config["stft_hop_sizes"],
        stft_win_lengths=disc_config["stft_win_lengths"],
    )


def build_mel_loss(config):
    audio_config = config["audio"]
    return MultiScaleMelSpectrogramLoss(
        sample_rate=audio_config["sample_rate"],
        n_mels=audio_config["n_mels"],
        fft_sizes=audio_config["mel_fft_sizes"],
        hop_sizes=audio_config["mel_hop_sizes"],
        win_lengths=audio_config["mel_win_lengths"],
        f_min=audio_config["mel_fmin"],
        f_max=audio_config["mel_fmax"],
    )


def build_train_loader(config):
    dataset = LibriSpeechTrainDataset(
        root=config["data"]["train_root"],
        sample_rate=config["audio"]["sample_rate"],
        crop_seconds=config["audio"]["crop_seconds"],
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_audio,
    )


def build_eval_loader(config):
    dataset = FullAudioDataset(
        root=config["data"]["valid_root"],
        sample_rate=config["audio"]["sample_rate"],
        max_files=config["data"].get("eval_num_files"),
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
