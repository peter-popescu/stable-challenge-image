"""Read-only training resource metrics, persisted across milestone resumes."""
import json
from pathlib import Path
import time
import lightning as pl
import torch


class ResourceMetrics(pl.Callback):
    def __init__(self, output):
        self.path = Path(output) / 'resources.json'
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'epochs': {}}

    def on_train_epoch_start(self, trainer, module):
        if module.device.type == 'cuda':
            torch.cuda.synchronize(module.device)
        self.start = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if module.device.type == 'cuda' and batch_idx % 10 == 0:
            for name, value in [('gpu_memory_allocated_gib', torch.cuda.memory_allocated(module.device)/1024**3),
                                ('gpu_memory_reserved_gib', torch.cuda.memory_reserved(module.device)/1024**3),
                                ('gpu_memory_peak_gib', torch.cuda.max_memory_allocated(module.device)/1024**3)]:
                module.log(name, value, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer, module):
        if module.device.type == 'cuda':
            torch.cuda.synchronize(module.device)
            self.data.update(gpu=torch.cuda.get_device_name(module.device),
                             peak_allocated_gib=max(self.data.get('peak_allocated_gib', 0), torch.cuda.max_memory_allocated(module.device)/1024**3),
                             peak_reserved_gib=max(self.data.get('peak_reserved_gib', 0), torch.cuda.max_memory_reserved(module.device)/1024**3))
        seconds = time.perf_counter()-self.start
        images = trainer.num_training_batches * module.config['batch_size']
        self.data['epochs'][str(trainer.current_epoch+1)] = dict(seconds=seconds, images=images, images_per_second=images/seconds)
        self.data['training_wall_seconds'] = sum(r['seconds'] for r in self.data['epochs'].values())
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, indent=2)+'\n')
        temporary.replace(self.path)
        module.log('training_images_per_second', images/seconds, on_step=False, on_epoch=True)
