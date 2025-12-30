from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from .train import TrainingRunner

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_mode = str(getattr(cfg, "run_mode", "sample")).lower()
    if run_mode == "train":
        log.warning("run_mode=train supplied to sample.py; overriding to 'sample'.")
        run_mode = "sample"
    cfg.run_mode = run_mode

    if not getattr(cfg, "checkpoint_path", None):
        raise ValueError("checkpoint_path must be provided when invoking sample.py.")

    runner = TrainingRunner(cfg)
    if runner.run_mode == "train":
        raise ValueError("TrainingRunner resolved run_mode='train'. Set run_mode='sample'.")

    runner.sample()


if __name__ == "__main__":
    main()
