import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from soundstream.config import load_config
from soundstream.metrics import MetricTracker, average_metrics
from soundstream.model import build_eval_loader, build_mel_loss, build_model


def main():
    config = load_config("configs/config.yaml")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config["eval"]["checkpoint_path"], map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mel_loss_fn = build_mel_loss(config).to(device)
    tracker = MetricTracker(config["audio"]["sample_rate"], use_nisqa=config["eval"]["use_nisqa"])
    eval_loader = build_eval_loader(config)

    metrics = []
    mel_losses = []
    with torch.no_grad():
        for audio, name in tqdm(eval_loader, desc="Evaluating"):
            audio = audio.to(device)
            output = model(audio)
            reconstructed = output["audio"][..., : audio.size(-1)]
            mel_losses.append(float(mel_loss_fn(reconstructed, audio).item()))
            item = tracker.update(reconstructed, audio)
            item["file"] = name[0]
            metrics.append(item)

    mean_metrics = average_metrics([{key: value for key, value in item.items() if key != "file"} for item in metrics])
    mean_metrics["mel"] = sum(mel_losses) / len(mel_losses)
    with Path(config["eval"]["output_path"]).open("w", encoding="utf-8") as file:
        json.dump({"mean": mean_metrics, "items": metrics}, file, indent=2, ensure_ascii=True)
    print(mean_metrics)


if __name__ == "__main__":
    main()
