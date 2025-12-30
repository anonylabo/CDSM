from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Set

from pytorch_lightning import Callback, LightningModule, Trainer


class PeriodicCheckpoint(Callback):
    """Save checkpoints at fixed epoch intervals and/or at specific epochs.

    Creates a dedicated directory (e.g., checkpoints_periodic) under the run folder,
    with files named using a pattern like epoch{epoch:03d}.ckpt.
    Epoch numbers are treated as 1-based.
    """

    def __init__(
        self,
        dirpath: Path | str,
        every_n_epochs: Optional[int] = None,
        epochs: Optional[Sequence[int]] = None,
        filename_pattern: str = "epoch{epoch:03d}.ckpt",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = int(every_n_epochs) if every_n_epochs else None
        self.epochs: Set[int] = set(int(e) for e in epochs) if epochs else set()
        self.filename_pattern = str(filename_pattern)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch_num = int(trainer.current_epoch) + 1  # convert to 1-based
        save = False
        if self.every_n_epochs and self.every_n_epochs > 0 and epoch_num % self.every_n_epochs == 0:
            save = True
        if self.epochs and epoch_num in self.epochs:
            save = True
        if not save:
            return
        filename = self.filename_pattern.format(epoch=epoch_num)
        ckpt_path = self.dirpath / filename
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(ckpt_path))

