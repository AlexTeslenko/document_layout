from typing import Dict

import yaml


def load_config(config_pth: str) -> Dict:
    with open(config_pth) as file:
        config = yaml.safe_load(file)

    return config
