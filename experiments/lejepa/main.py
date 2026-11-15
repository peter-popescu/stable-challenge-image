"""Offline Lightning training. Run with python -m experiments.lejepa.main."""
import argparse
import hashlib
import importlib.metadata
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import platform
import subprocess
import math
import shutil
import lightning as pl
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from .data import Views
from .model import make_method
from stable_pretraining.registry import RegistryLogger


class RetainedCheckpoint(ModelCheckpoint):
    """Keep milestone checkpoints with a state key distinct from the latest saver."""


def training_budget(config, stop_after=None):
    """Limit execution while preserving config['epochs'] as the schedule horizon."""
    budget = config['epochs'] if stop_after is None else stop_after
    if not 1 <= budget <= config['epochs']:
        raise ValueError('Execution budget must be between 1 and the configured schedule horizon')
    return budget


class EpochSeed(pl.Callback):
    """Epoch-boundary resumes reproduce shuffle and augmentation RNG streams."""
    def on_train_epoch_start(self, trainer, module):
        pl.seed_everything(module.config["seed"] + trainer.current_epoch, workers=True)


class Experiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters({'config': config})
        self.config = config
        self.method = make_method(config)

    def training_step(self, batch, batch_idx):
        result = self.method(global_views=batch['global'], local_views=batch['local'])
        if not torch.isfinite(result.loss):
            raise FloatingPointError('SSL loss became nonfinite')
        for name, value in [('loss', result.loss), ('invariance', result.inv_loss),
                            ('sigreg', result.sigreg_loss)]:
            self.log(name, value, on_step=True, on_epoch=True,
                     batch_size=batch['global'][0].shape[0])
        return result.loss

    def configure_optimizers(self):
        c = self.config
        opt = torch.optim.AdamW(self.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        def scale(epoch):
            if epoch < c['warmup_epochs']:
                return (epoch + 1) / max(1, c['warmup_epochs'])
            progress = (epoch - c['warmup_epochs']) / max(1, c['epochs'] - c['warmup_epochs'])
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
        return [opt], [torch.optim.lr_scheduler.LambdaLR(opt, scale)]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after-epochs', type=int, help='Execution limit; does not change the configured LR schedule')
    p.add_argument('--retain-every', type=int, default=0, help='Retain additional milestone checkpoints every N epochs')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Training (including smoke) must run inside a Slurm allocation on this cluster')
    c = json.loads(a.config.read_text())
    if a.smoke:
        c.update(epochs=1, batch_size=2, workers=0, n_slices=8, warmup_epochs=0)
    if c['global_views'] < 1 or c['local_views'] < 1 or c['batch_size'] < 2:
        p.error('Require global/local views and batch_size >= 2')
    try:
        budget = training_budget(c, a.stop_after_epochs)
    except ValueError as error:
        p.error(str(error))
    if a.retain_every < 0:
        p.error('--retain-every must be nonnegative')
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)['hyper_parameters']['config']
        if saved != c:
            p.error('Resume requires the original resolved configuration')
    pl.seed_everything(c['seed'], workers=True)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        source = a.output / 'source'
        source.mkdir(exist_ok=True)
        shutil.copytree(Path(__file__).parent, source / 'lejepa', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        for filename in ['pyproject.toml', 'uv.lock', 'training_data.py', 'export_onnx.py']:
            shutil.copy2(filename, source / filename)
    dataset = Views(str(a.data), c)
    if len(dataset) < c['batch_size']:
        p.error('Dataset smaller than batch size; drop_last would yield no batches')
    digest = hashlib.sha256()
    with a.data.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = {'config': c, 'data': str(a.data.resolve()), 'data_sha256': digest.hexdigest(), 'dataset_size': len(dataset),
                'python': platform.python_version(), 'torch': torch.__version__,
                'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_diff': subprocess.check_output(['git', 'diff'], text=True),
                'resume': str(a.resume),
                'runtime': {'stop_after_epochs': a.stop_after_epochs, 'max_epochs': budget,
                            'retain_every': a.retain_every}}
    (a.output / ('resume.json' if a.resume else 'resolved.json')).write_text(json.dumps(metadata, indent=2))
    loader = DataLoader(dataset, batch_size=c['batch_size'], shuffle=True,
                        num_workers=c['workers'], drop_last=True)
    callbacks = [EpochSeed(), LearningRateMonitor(logging_interval='step'),
                 ModelCheckpoint(dirpath=a.output / 'checkpoints', save_last=True,
                                 every_n_epochs=1, save_top_k=1, monitor=None)]
    if a.retain_every:
        callbacks.append(RetainedCheckpoint(
            dirpath=a.output / 'checkpoints', filename='retained-epoch-{epoch:03d}-step-{step}',
            auto_insert_metric_name=False, every_n_epochs=a.retain_every,
            save_top_k=-1, save_last=False, monitor=None))
    trainer = pl.Trainer(accelerator=a.accelerator, devices=1, max_epochs=budget,
                         deterministic=True, precision='32-true',
                         limit_train_batches=2 if a.smoke else 1.0,
                         logger=RegistryLogger(
                             run_dir=a.output / 'metrics',
                             run_id=str(a.output.resolve().relative_to(Path.cwd().resolve()))
                             if a.output.resolve().is_relative_to(Path.cwd().resolve())
                             else str(a.output.resolve()),
                             tags=['lejepa', 'resnet9', 'smoke' if a.smoke else 'training'],
                             flush_logs_every_n_steps=10), log_every_n_steps=1,
                         callbacks=callbacks)
    trainer.fit(Experiment(c), loader, ckpt_path=str(a.resume) if a.resume else None)


if __name__ == '__main__':
    main()
