from pathlib import Path
from urllib.request import urlretrieve

import yaml


def main():
    with Path("configs/config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    output_path = Path(config["checkpoint_download"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(config["checkpoint_download"]["url"], output_path)
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
