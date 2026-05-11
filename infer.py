import sys
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from soundstream.config import load_config
from soundstream.data import load_audio
from soundstream.model import build_model


def main():
    config = load_config("configs/config.yaml")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config["infer"]["checkpoint_path"], map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    audio = load_audio(config["infer"]["input_path"], config["audio"]["sample_rate"]).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(audio)["audio"][..., : audio.size(-1)].cpu().squeeze(0)
    torchaudio.save(config["infer"]["output_path"], reconstructed, config["audio"]["sample_rate"])


if __name__ == "__main__":
    main()
