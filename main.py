from trainer import QGTrainer, DualTrainer, QATrainer
from infenrence import BeamSearcher
import config


def main():
    if config.train:
        # trainer = DualTrainer(config.qa_path, config.ca2q_path, config.c2q_path, config.c2a_path)
        # trainer = QGTrainer()
        trainer = QATrainer()
        trainer.train()
    else:
        print("start decoding")
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    main()
