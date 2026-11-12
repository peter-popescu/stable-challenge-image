"""Local DINO experiment using stable-pretraining; see DINO.md for commands."""

import argparse
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.dino import DINO
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from backbones import SmallResNet18
from export_onnx import MEAN, STD
from probe import ChallengeProbe
from submission import export_teacher
from training_data import DEFAULT_PATH, load_training_data


def crop_transform(size, scale, blur_probability, solarize=False):
    operations = [
        T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=blur_probability),
    ]
    if solarize:
        operations.append(T.RandomSolarize(128, p=0.2))
    return T.Compose([*operations, T.ToTensor(), T.Normalize(MEAN, STD)])


class DINOCrops(Dataset):
    """Read only images from local parquet; never pass labels to the model."""

    def __init__(self, path, local_crops=6, limit=None):
        self.images = load_training_data(str(path)).select_columns(["image"])
        if limit is not None:
            self.images = self.images.select(range(min(limit, len(self.images))))
        self.global_transforms = [
            crop_transform(96, (0.4, 1.0), 1.0),
            crop_transform(96, (0.4, 1.0), 0.1, solarize=True),
        ]
        self.local_transform = crop_transform(48, (0.05, 0.4), 0.5)
        self.local_crops = local_crops

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]["image"].convert("RGB")
        return {
            "global_views": [transform(image) for transform in self.global_transforms],
            "local_views": [self.local_transform(image) for _ in range(self.local_crops)],
        }


def dino_schedule(optimizer, module):
    total_steps = int(module.trainer.estimated_stepping_batches)
    if total_steps < 2:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    return LinearWarmupCosineAnnealing(
        optimizer, total_steps=total_steps,
        peak_step=max(1, int(0.1 * total_steps)),
    )


class NormalizedPrototypes(torch.nn.Linear):
    """Unit-norm prototype weights, equivalent to weight norm with fixed g=1."""

    def forward(self, x):
        return torch.nn.functional.linear(x, torch.nn.functional.normalize(self.weight, dim=1))


class CNNTeacherStudentWrapper(spt.TeacherStudentWrapper):
    """Average floating state, but copy integer BatchNorm counters exactly."""

    @torch.no_grad()
    def update_teacher(self):
        super().update_teacher()
        if self.training and self.ema_coefficient.item() != 1:
            for teacher, student in zip(self.teacher.buffers(), self.student.buffers()):
                if not teacher.is_floating_point():
                    teacher.copy_(student)


class ChallengeDINO(DINO):
    """Adapt stable-pretraining's DINO method to its dictionary training loop."""

    def __init__(self, epochs=100, lr=5e-4, weight_decay=0.04, n_prototypes=4096,
                 normalize_prototypes=False, freeze_last_layer_epochs=0,
                 backbone="vit_tiny"):
        # The constructor default keeps checkpoints without a backbone field
        # compatible. New CLI runs select ResNet below.
        if backbone == "vit_tiny":
            encoder = "vit_tiny_patch16_224"
            encoder_kwargs = {"img_size": 96, "patch_size": 8, "depth": 10,
                              "dynamic_img_size": True}
        elif backbone == "resnet18_small":
            encoder = SmallResNet18().eval()
            encoder_kwargs = None
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        super().__init__(
            encoder_name=encoder,
            encoder_kwargs=encoder_kwargs,
            pretrained=False,
            projector_hidden_dim=1024,
            projector_bottleneck_dim=128,
            n_prototypes=n_prototypes,
            warmup_epochs_temperature_teacher=min(30, max(1, epochs // 3)),
        )
        if backbone == "resnet18_small":
            self.backbone = CNNTeacherStudentWrapper(self.backbone.student)
            # DINO's dummy shape inference must not update BatchNorm stats;
            # restore training mode afterwards, including all child modules.
            self.backbone.train(self.training)
        self.save_hyperparameters({
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "n_prototypes": n_prototypes,
            "normalize_prototypes": normalize_prototypes,
            "freeze_last_layer_epochs": freeze_last_layer_epochs,
            "backbone": backbone,
        })
        # Defaults retain older checkpoint semantics; new runs enable both below.
        if normalize_prototypes:
            projector = self.projector.student
            projector[-1] = NormalizedPrototypes(128, n_prototypes, bias=False)
            for layer in projector.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
            self.projector = spt.TeacherStudentWrapper(projector)
        # In 0.1.8 the loss center is a plain attribute. Register it so device
        # transfers and checkpoints preserve this essential part of DINO state.
        del self.dino_loss.center
        self.dino_loss.register_buffer("center", torch.zeros(1, n_prototypes))
        self.optim = {
            "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": weight_decay,
                          "exclude_bias_norm": True},
            "scheduler": dino_schedule,
            "interval": "step",
            "gradient_clip_val": 3.0,
        }

    def forward(self, batch, stage="fit"):
        output = super().forward(
            global_views=batch["global_views"], local_views=batch["local_views"]
        )
        # Finish the queued center update so every checkpoint has current state.
        self.dino_loss.apply_center_update()
        if not torch.isfinite(output.loss):
            raise FloatingPointError("DINO loss became non-finite")
        self.log("train/loss", output.loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch["global_views"][0].shape[0])
        return {"loss": output.loss, "embedding": output.embedding}

    def after_manual_backward(self):
        # stable-pretraining calls this before clipping and optimizer.step().
        if self.current_epoch < self.hparams.freeze_last_layer_epochs:
            for parameter in self.projector.student[-1].parameters():
                parameter.grad = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_PATH))
    parser.add_argument("--backbone", choices=["resnet18_small", "vit_tiny"],
                        help="New-run encoder (default: resnet18_small); resume restores saved architecture")
    parser.add_argument("--output", type=Path, default=Path("runs/dino"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--local-crops", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04,
                        help="AdamW weight decay for new runs (resume restores the saved value)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-every", type=int, default=5,
                        help="Log challenge-protocol leave-one-out accuracy every N epochs; 0 disables")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--wandb", action="store_true", help="Log metrics and configuration to Weights & Biases")
    parser.add_argument("--wandb-project", default="dino-experiment")
    parser.add_argument("--wandb-entity", help="W&B username or team (defaults to your account)")
    parser.add_argument("--wandb-name", help="Display name for this training run")
    parser.add_argument("--wandb-id", help="Existing W&B run ID to continue alongside --resume")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--resume", type=Path, help="Resume a trusted local training checkpoint")
    parser.add_argument("--export-checkpoint", type=Path, help="Export a trusted local checkpoint without training")
    parser.add_argument("--smoke-test", action="store_true", help="Two small CPU training steps, checkpoint, and ONNX export")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.local_crops < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; workers and local-crops nonnegative")
    if args.resume and args.export_checkpoint:
        parser.error("choose either --resume or --export-checkpoint")
    if not 0 <= args.weight_decay < float("inf"):
        parser.error("weight-decay must be finite and nonnegative")
    if args.probe_every < 0:
        parser.error("probe-every must be nonnegative")
    return args


def main():
    args = parse_args()
    spt.set(cache_dir=str((args.output / "spt-cache").resolve()))
    pl.seed_everything(args.seed, workers=True)
    if args.export_checkpoint:
        model = ChallengeDINO.load_from_checkpoint(str(args.export_checkpoint), map_location="cpu")
        export_teacher(model, args.output / "model.onnx")
        return
    if args.smoke_test:
        args.accelerator, args.precision = "cpu", "32-true"
        args.batch_size, args.workers, args.local_crops = 2, 0, 2
        torch.set_num_threads(2)
    dataset = DINOCrops(args.data, args.local_crops, limit=4 if args.smoke_test else None)
    if len(dataset) < args.batch_size:
        raise ValueError("Dataset must contain at least one full batch; lower --batch-size")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    if args.resume:
        model = ChallengeDINO.load_from_checkpoint(str(args.resume), map_location="cpu")
        if args.backbone is not None and args.backbone != model.hparams.backbone:
            raise ValueError("--backbone must match the resumed checkpoint")
    else:
        model = ChallengeDINO(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              normalize_prototypes=True, freeze_last_layer_epochs=1,
                              backbone=args.backbone or "resnet18_small")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", save_last=True,
                                 save_top_k=0)
    callbacks = [TeacherStudentCallback()]
    if args.probe_every:
        callbacks.append(ChallengeProbe(args.data, every_n_epochs=args.probe_every))
    callbacks.append(checkpoint)
    loggers = [CSVLogger(str(args.output), name="logs")]
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name, id=args.wandb_id,
            save_dir=str(args.output), mode=args.wandb_mode,
            log_model=False,
            config={
                "encoder": model.hparams.backbone, "embedding_dim": model.embed_dim,
                "pretrained": False,
                "batch_size": args.batch_size, "local_crops": args.local_crops,
                "global_crop_size": 96, "local_crop_size": 48,
                "seed": args.seed, "precision": args.precision,
                "smoke_test": args.smoke_test,
                "probe_every": args.probe_every, "probe_protocol": "onnx_cpu_ridge_loo",
            },
        )
        loggers.append(wandb_logger)
    trainer = pl.Trainer(
        max_epochs=args.epochs, max_steps=2 if args.smoke_test else -1,
        accelerator=args.accelerator, devices=1, precision=args.precision,
        # Multi-crop ViTs resize positional embeddings with bicubic interpolation.
        # Its CUDA backward lacks a deterministic implementation: warn rather
        # than abort while still preferring deterministic kernels where possible.
        deterministic="warn", logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1 if args.smoke_test else 10,
        enable_model_summary=False,
    )
    exit_code = 1
    try:
        trainer.fit(model, train_dataloaders=loader,
                    ckpt_path=str(args.resume) if args.resume else None)
        trainer.save_checkpoint(args.output / "checkpoints" / "last.ckpt")
        export_teacher(model, args.output / "model.onnx")
        exit_code = 0
    finally:
        if wandb_logger is not None:
            import wandb

            if wandb.run is not None:
                wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
