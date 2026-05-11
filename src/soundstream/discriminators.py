import torch
from torch import nn


def _activation():
    return nn.LeakyReLU(0.2)


class WaveDiscriminator(nn.Module):
    def __init__(self, channels, use_spectral_norm=False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        channel_schedule = [1, channels, channels * 4, channels * 16, channels * 64, channels * 64, channels * 64]
        kernel_sizes = [15, 41, 41, 41, 41, 5]
        strides = [1, 4, 4, 4, 4, 1]
        groups = [1, 4, 16, 64, 256, 1]

        blocks = []
        for index in range(len(kernel_sizes)):
            blocks.append(
                norm(
                    nn.Conv1d(
                        channel_schedule[index],
                        channel_schedule[index + 1],
                        kernel_size=kernel_sizes[index],
                        stride=strides[index],
                        padding=(kernel_sizes[index] - 1) // 2,
                        groups=groups[index],
                    )
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final = norm(nn.Conv1d(channel_schedule[-1], 1, kernel_size=3, padding=1))

    def forward(self, audio):
        features = []
        x = audio
        for block in self.blocks:
            x = _activation()(block(x))
            features.append(x)
        logits = self.final(x)
        return logits, features


class MultiScaleWaveDiscriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                WaveDiscriminator(channels, use_spectral_norm=True),
                WaveDiscriminator(channels),
                WaveDiscriminator(channels),
            ]
        )
        self.pooling = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            ]
        )

    def forward(self, audio):
        outputs = []
        x = audio
        for index, discriminator in enumerate(self.discriminators):
            if index > 0:
                x = self.pooling[index](x)
            outputs.append(discriminator(x))
        return outputs


class STFTSubDiscriminator(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        channels = [in_channels, base_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        strides = [(1, 2), (2, 2), (1, 2), (2, 2), (1, 2)]
        modules = []
        for index, stride in enumerate(strides):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        kernel_size=(stride[0] + 2, stride[1] + 2),
                        stride=stride,
                        padding=1,
                    ),
                    _activation(),
                )
            )
        self.blocks = nn.ModuleList(modules)
        self.final = nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        logits = self.final(x)
        return logits, features


class MultiResolutionSTFTDiscriminator(nn.Module):
    def __init__(self, channels, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.discriminators = nn.ModuleList([STFTSubDiscriminator(2, channels) for _ in fft_sizes])

    @staticmethod
    def _stft_representation(audio, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length, device=audio.device)
        stft = torch.stft(
            audio.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        return torch.stack([stft.real, stft.imag], dim=1)

    def forward(self, audio):
        outputs = []
        for discriminator, n_fft, hop_length, win_length in zip(
            self.discriminators,
            self.fft_sizes,
            self.hop_sizes,
            self.win_lengths,
        ):
            stft = self._stft_representation(audio, n_fft, hop_length, win_length)
            outputs.append(discriminator(stft))
        return outputs


class SoundStreamDiscriminator(nn.Module):
    def __init__(
        self,
        wave_channels,
        stft_channels,
        stft_fft_sizes,
        stft_hop_sizes,
        stft_win_lengths,
    ):
        super().__init__()
        self.wave = MultiScaleWaveDiscriminator(wave_channels)
        self.stft = MultiResolutionSTFTDiscriminator(
            channels=stft_channels,
            fft_sizes=stft_fft_sizes,
            hop_sizes=stft_hop_sizes,
            win_lengths=stft_win_lengths,
        )

    def forward(self, audio):
        return {
            "wave": self.wave(audio),
            "stft": self.stft(audio),
        }
