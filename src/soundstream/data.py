import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def list_audio_files(root):
    root_path = Path(root)
    return sorted(path for path in root_path.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def load_audio(path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def random_crop_with_repetition(audio, target_length):
    length = audio.size(-1)
    if length >= target_length:
        start = random.randint(0, length - target_length)
        return audio[..., start : start + target_length]

    repeat_count = (target_length + length - 1) // length
    padded = audio.repeat(1, repeat_count)[..., :target_length]
    return padded


class LibriSpeechTrainDataset(Dataset):
    def __init__(self, root, sample_rate, crop_seconds):
        self.files = list_audio_files(root)
        self.sample_rate = sample_rate
        self.crop_length = int(sample_rate * crop_seconds)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio = load_audio(self.files[index], self.sample_rate)
        audio = random_crop_with_repetition(audio, self.crop_length)
        return audio.clamp(-1.0, 1.0)


class FullAudioDataset(Dataset):
    def __init__(self, root, sample_rate, max_files=None):
        files = list_audio_files(root)
        self.files = files[:max_files] if max_files is not None else files
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        audio = load_audio(path, self.sample_rate).clamp(-1.0, 1.0)
        return audio, path.name


def collate_audio(batch):
    return torch.stack(batch, dim=0)
