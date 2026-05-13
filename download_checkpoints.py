from pathlib import Path

import gdown
import yaml


def main():
    with Path("configs/config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    output_path = Path(config["checkpoint_download"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(config["checkpoint_download"]["url"], str(output_path), quiet=False, fuzzy=True)
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
