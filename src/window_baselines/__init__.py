"""Causal window-space baselines aligned with the main diffusion experiments.

These baselines live outside src/baselines/ so they remain available even if
that package is removed. They reuse the same datamodule/config style as the
main experiments and emit outputs in the familiar per-window CSV/PT plus
summary JSON format under ``outputs/baselines/<method>/conditional/<timestamp>``.
"""

# Re-export the runner for convenience.
from window_baselines.cmd.run import main  # noqa: F401

