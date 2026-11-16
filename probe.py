"""Periodic challenge-protocol monitoring on labelled training examples."""

import os
import json
import shutil
import tempfile
import time
from pathlib import Path

import lightning as pl

from evaluate import evaluate
from diagnose import diagnose
from submission import export_teacher


class ChallengeProbe(pl.Callback):
    """Evaluate a CPU ONNX export with the challenge's exact ridge/LOO code."""

    def __init__(self, data, every_n_epochs=5, support_data=None, output_dir=None,
                 low_shot=0, repeats=10, seed=2026):
        if every_n_epochs < 1:
            raise ValueError("Probe interval must be positive")
        self.data = str(data)
        self.every_n_epochs = every_n_epochs
        self.support_data = str(support_data) if support_data else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.best_score = float("-inf")
        if low_shot < 0 or repeats < 1:
            raise ValueError("low_shot must be nonnegative and repeats positive")
        self.low_shot, self.repeats, self.seed = low_shot, repeats, seed

    def on_fit_start(self, trainer, pl_module):
        # A continuation in a new output directory tracks its own best export.
        if self.output_dir and (self.output_dir / "best.onnx").exists():
            metadata_path = self.output_dir / "best.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata["data"] != self.data or metadata.get("support_data") != self.support_data:
                    raise ValueError("Existing best export used different monitoring data; use a new output directory")
                self.best_score = metadata["score"]

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch != 1 and epoch % self.every_n_epochs and epoch != trainer.max_epochs:
            return
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="dino-probe-") as directory:
            path = export_teacher(pl_module, Path(directory) / "model.onnx")
            threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
            if self.support_data:
                extra = {}
                if epoch == trainer.max_epochs and self.low_shot:
                    extra = {"low_shot": self.low_shot, "repeats": self.repeats, "seed": self.seed,
                             "embeddings_out": self.output_dir / "probe-final-embeddings.npz"
                             if self.output_dir else None}
                result = diagnose(str(path), self.data, support_data=self.support_data,
                                  threads=threads, **extra)
            else:
                result = evaluate(str(path), data=self.data, threads=threads)
            if result["score"] > self.best_score:
                self.best_score = result["score"]
                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    temporary = self.output_dir / "best.onnx.tmp"
                    shutil.copyfile(path, temporary)
                    temporary.replace(self.output_dir / "best.onnx")
                    metadata = {"score": self.best_score, "epoch": epoch,
                                "global_step": trainer.global_step, "data": self.data,
                                "support_data": self.support_data,
                                "protocol": "support_query" if self.support_data else "loo"}
                    temporary = self.output_dir / "best.json.tmp"
                    temporary.write_text(json.dumps(metadata, indent=2))
                    temporary.replace(self.output_dir / "best.json")
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report = {**result, "epoch": epoch, "global_step": trainer.global_step,
                      "selection_note": "Compare final accuracy; best accuracy is exploratory"}
            with (self.output_dir / "probe-history.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            if epoch == trainer.max_epochs:
                (self.output_dir / "probe-final.json").write_text(json.dumps(report, indent=2))
        prefix = "holdout" if self.support_data else "probe"
        metrics = {
            f"{prefix}/{'accuracy' if self.support_data else 'loo_accuracy'}": result["score"],
            f"{prefix}/best_accuracy": self.best_score,
            f"{prefix}/images": result["images"],
            f"{prefix}/classes": result["classes"],
            f"{prefix}/ridge_lambda": result["lambda"],
            f"{prefix}/seconds": time.perf_counter() - started,
            "epoch": trainer.current_epoch,
        }
        if self.support_data:
            metrics["holdout/macro_accuracy"] = result["macro_accuracy"]
        for name, domain in result.get("domains", {}).items():
            metrics[f"{prefix}/domain/{name}_accuracy"] = domain["accuracy"]
        if epoch == trainer.max_epochs:
            metrics[f"{prefix}/final_accuracy"] = result["score"]
            if self.support_data:
                metrics["holdout/final_macro_accuracy"] = result["macro_accuracy"]
        if "low_support" in result:
            low = result["low_support"]
            for key in ("accuracy_mean", "accuracy_std", "macro_accuracy_mean"):
                metrics[f"{prefix}/low_shot_{low['shots']}/{key}"] = low[key]
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)
        description = "held-out image accuracy" if self.support_data else "challenge-protocol LOO accuracy"
        print(f"Epoch {epoch}: {description} "
              f"{result['score']:.1%} ({result['images']} labelled monitoring images)",
              flush=True)
