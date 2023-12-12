from trainer import Training
from utils import load_config

def main() -> None:
    config = load_config(
        "configs/config.yaml"
    )
    training = Training(config)
    training.setup()
    training.start()
    training.eval()


if __name__ == "__main__":
    main()