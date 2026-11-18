"""Audit recorded optimizer LRs without importing Torch or loading checkpoints."""
import csv
import math
from pathlib import Path

from .sweep import digest


def verify_lr_history(screen, continued, config, dataset_size):
    steps_per_epoch = dataset_size // config['batch_size']
    if steps_per_epoch < 1:
        raise ValueError('Dataset must contain a full batch')
    rates, segments = {}, []
    max_error = 0.0
    for path in [Path(screen), Path(continued)]:
        steps = []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if not row.get('lr-AdamW'):
                    continue
                step = int(row['step'])
                if step in rates:
                    raise ValueError('Duplicate LR step across control segments')
                epoch = step // steps_per_epoch
                warmup = config['warmup_epochs']
                scale = ((epoch + 1) / max(1, warmup) if epoch < warmup else
                         .5 * (1 + math.cos(math.pi * min((epoch - warmup) /
                                                       max(1, config['epochs'] - warmup), 1))))
                expected = config['lr'] * scale
                actual = float(row['lr-AdamW'])
                if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError(f'Control LR differs from planned schedule at step {step}: {actual} != {expected}')
                rates[step] = actual
                steps.append(step)
                max_error = max(max_error, abs(actual - expected))
        if not steps or steps != list(range(steps[0], steps[-1] + 1)):
            raise ValueError('Incomplete or unordered LR segment')
        segments.append(dict(path=str(path), sha256=digest(path), rows=len(steps),
                             first_step=steps[0], last_step=steps[-1]))
    if sorted(rates) != list(range(config['epochs'] * steps_per_epoch)):
        raise ValueError('Control LR history does not cover the full configured schedule')
    if segments[0]['last_step'] != 100 * steps_per_epoch - 1 or segments[1]['first_step'] != 100 * steps_per_epoch:
        raise ValueError('Control continuation boundary differs from epoch 100')
    return dict(verdict='matched_200_epoch_schedule_from_initialization',
                schedule_epochs=config['epochs'], warmup_epochs=config['warmup_epochs'],
                steps_per_epoch=steps_per_epoch, checked_steps=len(rates),
                maximum_absolute_error=max_error, segments=segments,
                epoch_index_to_lr={str(epoch): rates[epoch * steps_per_epoch]
                                   for epoch in [0, 9, 10, 49, 99, 100, 149, 199]})
