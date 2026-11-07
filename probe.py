"""Periodic challenge-protocol monitoring on labelled training examples."""

import os
import tempfile
import time
from pathlib import Path

import lightning as pl

from evaluate import evaluate
from submission import export_teacher


class ChallengeProbe(pl.Callback):
    """Evaluate a CPU ONNX export with the challenge's exact ridge/LOO code."""

    def __init__(self, data, every_n_epochs=5):
        if every_n_epochs < 1:
            raise ValueError("Probe interval must be positive")
        self.data = str(data)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch != 1 and epoch % self.every_n_epochs and epoch != trainer.max_epochs:
            return
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="dino-probe-") as directory:
            path = export_teacher(pl_module, Path(directory) / "model.onnx")
            threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
            result = evaluate(str(path), data=self.data, threads=threads)
        metrics = {
            "probe/loo_accuracy": result["score"],
            "probe/images": result["images"],
            "probe/classes": result["classes"],
            "probe/ridge_lambda": result["lambda"],
            "probe/seconds": time.perf_counter() - started,
            "epoch": trainer.current_epoch,
        }
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)
        print(f"Epoch {epoch}: challenge-protocol LOO accuracy "
              f"{result['score']:.1%} ({result['images']} labelled training images)",
              flush=True)
