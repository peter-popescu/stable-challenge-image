"""Local DINO experiment using stable-pretraining; see DINO.md for commands."""

import argparse
import copy
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

from export_onnx import MEAN, STD, export_onnx
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


class ChallengeDINO(DINO):
    """Adapt stable-pretraining's DINO method to its dictionary training loop."""

    def __init__(self, epochs=100, lr=5e-4, weight_decay=0.04, n_prototypes=4096):
        super().__init__(
            encoder_name="vit_tiny_patch16_224",
            encoder_kwargs={
                "img_size": 96,
                "patch_size": 8,
                "depth": 10,
                "dynamic_img_size": True,
            },
            pretrained=False,
            projector_hidden_dim=1024,
            projector_bottleneck_dim=128,
            n_prototypes=n_prototypes,
            warmup_epochs_temperature_teacher=min(30, max(1, epochs // 3)),
        )
        self.save_hyperparameters({
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "n_prototypes": n_prototypes,
        })
        # In 0.1.8 the loss center is a plain attribute. Register it so device
        # transfers and checkpoints preserve this essential part of DINO state.
        del self.dino_loss.center
        self.dino_loss.register_buffer("center", torch.zeros(1, n_prototypes))
        self.optim = {
            "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": weight_decay},
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


def export_teacher(model, path):
    """Export only the EMA encoder; training heads stay out of the submission."""
    encoder = copy.deepcopy(model.backbone.teacher).cpu().eval()
    # Fixed spatial size makes the submitted graph simpler. Batch stays dynamic.
    encoder.dynamic_img_size = False
    encoder.patch_embed.strict_img_size = True
    encoder.patch_embed.output_fmt = "NCHW"
    encoder.patch_embed.flatten = True
    for block in encoder.blocks:
        block.attn.fused_attn = False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(encoder, str(path))
    size = path.stat().st_size / 1024**2
    if size > 20:
        raise ValueError(f"ONNX file is {size:.2f} MiB; challenge limit is 20 MiB")
    print(f"EMA teacher encoder: {size:.2f} MiB. Validate with test_submission.py.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DEFAULT_PATH))
    parser.add_argument("--output", type=Path, default=Path("runs/dino"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--local-crops", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
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
    model = ChallengeDINO(epochs=args.epochs, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", save_last=True,
                                 save_top_k=0)
    loggers = [CSVLogger(str(args.output), name="logs")]
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name, id=args.wandb_id,
            save_dir=str(args.output), mode=args.wandb_mode,
            log_model=False,
            config={
                "encoder": "vit_tiny_patch8_depth10", "pretrained": False,
                "batch_size": args.batch_size, "local_crops": args.local_crops,
                "global_crop_size": 96, "local_crop_size": 48,
                "seed": args.seed, "precision": args.precision,
                "smoke_test": args.smoke_test,
            },
        )
        loggers.append(wandb_logger)
    trainer = pl.Trainer(
        max_epochs=args.epochs, max_steps=2 if args.smoke_test else -1,
        accelerator=args.accelerator, devices=1, precision=args.precision,
        deterministic=True, logger=loggers,
        callbacks=[TeacherStudentCallback(), checkpoint],
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
