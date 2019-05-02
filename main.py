from trainer import Trainer, DualTrainer, QGTrainer, C2ATrainer
from infenrence import BeamSearcher
import config


def main():
    if config.train:
        trainer = QGTrainer()
        trainer.train()
    else:
        print("start decoding")
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    main()
