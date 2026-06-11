import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else Path(__file__).parent / "config.yaml"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}
