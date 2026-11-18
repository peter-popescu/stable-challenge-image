"""Protocol checks are stdlib-only; transform parity runs within Slurm."""
import copy
import json
import os
import csv
import math
import tempfile
from pathlib import Path
import unittest

from experiments.lejepa.augmentation_policy import resolve_augmentation
from experiments.lejepa.augmentation_study import load_matrix, canonical
from experiments.lejepa.lr_history import verify_lr_history


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path('experiments/lejepa/augmentations/study.json').read_text())

    def test_only_declared_interventions(self):
        runs = load_matrix(self.spec)
        base = runs[0]['config']
        self.assertEqual({k for k in base if base[k] != runs[1]['config'][k]}, {'local_scale'})
        self.assertEqual(runs[1]['name'], 'local-min10')
        self.assertEqual(runs[1]['config']['local_scale'], [.1, .6])
        self.assertEqual({k for k in base if base[k] != runs[2]['config'][k]},
                         {'jitter', 'jitter_contrast', 'jitter_saturation', 'jitter_hue', 'jitter_probability'})

    def test_optional_specs_are_isolated(self):
        for name, key, value in [('local-min50', 'local_scale', [.5, .6]),
                                 ('more-views', 'local_views', 6)]:
            spec = json.loads(Path(f'experiments/lejepa/augmentations/study-{name}.json').read_text())
            runs = load_matrix(spec)
            self.assertEqual([r['name'] for r in runs], ['control', name])
            self.assertEqual(runs[1]['config'][key], value)
            self.assertEqual({k for k in runs[0]['config'] if runs[0]['config'][k] != runs[1]['config'][k]}, {key})
        self.assertNotIn('more-views', [r['name'] for r in load_matrix(self.spec)])
        self.assertNotIn('local-min50', [r['name'] for r in load_matrix(self.spec)])

    def test_lr_history_detects_restarts_and_missing_steps(self):
        c = dict(lr=.003, warmup_epochs=10, epochs=200, batch_size=2)
        # Two optimizer steps/epoch; no Torch or checkpoint computation.
        rates = [.003*((e+1)/10 if e<10 else .5*(1+math.cos(math.pi*(e-10)/190)))
                 for e in range(200) for _ in range(2)]
        storage = Path(__file__).resolve().parents[1] / 'experiment_outputs/protocol-tests'
        storage.mkdir(parents=True, exist_ok=True)
        # Shared-filesystem cleanup can race with NFS visibility; cleanup failure
        # must not invalidate the protocol assertions or block GPU training.
        with tempfile.TemporaryDirectory(dir=storage, ignore_cleanup_errors=True) as temporary:
            paths = [Path(temporary) / name for name in ['screen.csv', 'continued.csv']]
            def write(values):
                for path, start in zip(paths, [0, 200]):
                    with path.open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=['step', 'lr-AdamW'])
                        writer.writeheader()
                        writer.writerows({'step': i, 'lr-AdamW': values[i]} for i in range(start, min(start+200, len(values))))
            write(rates)
            audit = verify_lr_history(*paths, c, 4)
            self.assertEqual(audit['checked_steps'], 400)
            write(rates[:200] + rates[:200])  # Warmup/cosine restarted at continuation.
            with self.assertRaises(ValueError):
                verify_lr_history(*paths, c, 4)
            write(rates[:-1])
            with self.assertRaises(ValueError):
                verify_lr_history(*paths, c, 4)
            # A screen actually scheduled for 100 epochs must also be rejected.
            short = [.003*((e+1)/10 if e<10 else .5*(1+math.cos(math.pi*(e-10)/90)))
                     for e in range(100) for _ in range(2)]
            write(short + rates[200:])
            with self.assertRaises(ValueError):
                verify_lr_history(*paths, c, 4)

    def test_historical_control_defaults(self):
        old = json.loads(Path('experiments/lejepa/config.json').read_text())
        old.update(lr=.003, lamb=.005)
        self.assertEqual(canonical(old), load_matrix(self.spec)[0]['config'])
        self.assertEqual(resolve_augmentation(old)['blur_probability'], 0)
        self.assertEqual(resolve_augmentation(old)['solarization_probability'], 0)

    def test_bad_ranges_rejected(self):
        base = load_matrix(self.spec)[0]['config']
        for change in [{'local_scale': [.7, .6]}, {'crop_ratio': [0, 1]},
                       {'grayscale_probability': 1.1}, {'blur_kernel': 2}, {'jitter_hue': .6}]:
            with self.assertRaises(ValueError):
                resolve_augmentation(dict(base, **change))

    def test_changed_schedule_or_matrix_rejected(self):
        changed = copy.deepcopy(self.spec)
        changed['milestones'] = [100, 200]
        with self.assertRaises(ValueError):
            load_matrix(changed)
        changed = copy.deepcopy(self.spec)
        changed['runs'][1]['name'] = 'another-policy'
        with self.assertRaises(ValueError):
            load_matrix(changed)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID'), 'Transform compute requires Slurm')
    def test_legacy_transform_rng_and_tensor_parity(self):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms as T
        from experiments.lejepa.data import augmentation, preprocessing
        c = json.loads(Path('experiments/lejepa/config.json').read_text())
        image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (96, 96, 3), dtype=np.uint8))
        for local in [False, True]:
            legacy = T.Compose([
                T.RandomResizedCrop(c['local_size'] if local else 96,
                                    scale=c['local_scale'] if local else c['global_scale'],
                                    ratio=(.9, 1.1), antialias=True),
                T.RandomHorizontalFlip(c['horizontal_flip']), T.RandomVerticalFlip(c['vertical_flip']),
                T.RandomApply([T.ColorJitter(c['jitter'], c['jitter'], c['jitter'], .02)],
                              p=c['jitter_probability']),
                T.RandomGrayscale(c['grayscale_probability']), preprocessing()])
            for seed in [0, 42, 99]:
                torch.manual_seed(seed)
                expected = legacy(image)
                expected_rng = torch.get_rng_state()
                torch.manual_seed(seed)
                actual = augmentation(c, local)(image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        all_runs = load_matrix(self.spec)
        for name in ['local-min50', 'more-views']:
            optional = json.loads(Path(f'experiments/lejepa/augmentations/study-{name}.json').read_text())
            all_runs += load_matrix(optional)[1:]
        for run in all_runs:
            for local, size in [(False, 96), (True, 48)]:
                tensor = augmentation(run['config'], local)(image)
                self.assertEqual(tuple(tensor.shape), (3, size, size))
                self.assertTrue(torch.isfinite(tensor).all())


if __name__ == '__main__':
    unittest.main()
