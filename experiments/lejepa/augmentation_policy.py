"""Resolve historical implicit augmentation defaults without importing Torch."""
import math


def resolve_augmentation(config):
    c = config
    policy = {k: c[k] for k in ('global_views', 'local_views', 'local_size',
                               'global_scale', 'local_scale', 'jitter',
                               'jitter_probability', 'grayscale_probability',
                               'horizontal_flip', 'vertical_flip')}
    defaults = dict(global_size=96, crop_ratio=[0.9, 1.1], jitter_hue=0.02,
                    jitter_contrast=c['jitter'], jitter_saturation=c['jitter'],
                    blur_probability=0.0, blur_kernel=7, blur_sigma=[0.1, 2.0],
                    solarization_probability=0.0, solarization_threshold=128)
    policy.update({k: c.get(k, v) for k, v in defaults.items()})
    for key in ('jitter_probability', 'grayscale_probability', 'horizontal_flip',
                'vertical_flip', 'blur_probability', 'solarization_probability'):
        if not 0 <= policy[key] <= 1:
            raise ValueError(f'{key} must be a probability')
    for key in ('global_scale', 'local_scale', 'crop_ratio', 'blur_sigma'):
        lo, hi = policy[key]
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
            raise ValueError(f'Invalid range: {key}')
        if key.endswith('_scale') and hi > 1:
            raise ValueError('Crop area fractions must not exceed one')
    for key in ('global_views', 'local_views', 'global_size', 'local_size', 'blur_kernel'):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if policy['blur_kernel'] % 2 != 1:
        raise ValueError('Blur kernel must be odd')
    for key in ('jitter', 'jitter_contrast', 'jitter_saturation', 'jitter_hue'):
        if not math.isfinite(policy[key]) or policy[key] < 0:
            raise ValueError(f'Invalid jitter magnitude: {key}')
    if policy['jitter_hue'] > 0.5 or not 0 <= policy['solarization_threshold'] <= 255:
        raise ValueError('Invalid hue or PIL solarization threshold')
    return policy


def augmentation_metadata(config):
    return dict(parameters=resolve_augmentation(config),
                order=['random_resized_crop', 'horizontal_flip', 'vertical_flip',
                       'random_apply_color_jitter', 'grayscale', 'optional_blur',
                       'optional_solarization', 'to_tensor', 'imagenet_normalize'],
                interpolation='bilinear', antialias=True, rotations='none',
                other_geometry='none', input='RGB PIL uint8',
                normalization_mean=[0.485, 0.456, 0.406],
                normalization_std=[0.229, 0.224, 0.225])
