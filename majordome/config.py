from pathlib import Path
import tomllib

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)

config = load_config()