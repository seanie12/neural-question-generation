from trainer import Trainer
from infenrence import BeamSearcher
import config


def main():
    if config.train:
        trainer = Trainer()
        trainer.train()
    else:
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    main()
