from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "ml/experiments/phase1/cleaned"
    output_dir: str = "ml/experiments/phase2"
    batch_size: int = 16
    epochs_freeze: int = 2
    epochs_finetune: int = 3
    lr: float = 1e-4
    patience: int = 3
    loss: str = "CrossEntropy"
