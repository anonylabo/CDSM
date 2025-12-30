from __future__ import annotations

import hydra
from omegaconf import DictConfig

from fdiff.cmd.train import TrainingRunner


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.run_mode = "sample"
    runner = TrainingRunner(cfg)
    runner.sample()


if __name__ == "__main__":
    main()
