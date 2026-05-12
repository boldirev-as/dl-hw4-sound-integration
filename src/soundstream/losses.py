import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate, n_mels, fft_sizes, hop_sizes, win_lengths, f_min, f_max):
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mels=n_mels,
                    f_min=f_min,
                    f_max=f_max,
                    power=1.0,
                )
                for n_fft, hop_length, win_length in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, predicted, target):
        loss = predicted.new_zeros(())
        for transform in self.transforms:
            pred_mel = transform(predicted.squeeze(1)).clamp_min(1e-5).log()
            target_mel = transform(target.squeeze(1)).clamp_min(1e-5).log()
            n = min(pred_mel.size(-1), target_mel.size(-1))
            pred_mel = pred_mel[..., :n]
            target_mel = target_mel[..., :n]
            loss = loss + F.l1_loss(pred_mel, target_mel)
        return loss / len(self.transforms)


def discriminator_loss(real_outputs, fake_outputs):
    real_outputs = list(real_outputs)
    fake_outputs = list(fake_outputs)
    loss = None
    for (real_logits, _), (fake_logits, _) in zip(real_outputs, fake_outputs):
        current = torch.mean((1.0 - real_logits) ** 2) + torch.mean(fake_logits**2)
        loss = current if loss is None else loss + current
    return loss / len(real_outputs)


def generator_adversarial_loss(fake_outputs):
    losses = [torch.mean((1.0 - fake_logits) ** 2) for fake_logits, _ in fake_outputs]
    return sum(losses) / len(losses)


def feature_matching_loss(real_outputs, fake_outputs):
    total = 0
    count = 0
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
        for real_feature, fake_feature in zip(real_features, fake_features):
            current = F.l1_loss(fake_feature, real_feature.detach())
            total = total + current
            count += 1
    return total / count
