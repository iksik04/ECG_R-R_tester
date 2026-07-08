import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config