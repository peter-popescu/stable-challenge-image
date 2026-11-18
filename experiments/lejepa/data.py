"""Image-only dataset: labels are removed before examples enter the loader."""
from torch.utils.data import Dataset
from torchvision import transforms as T
from training_data import load_training_data
from export_onnx import MEAN, STD
from .augmentation_policy import resolve_augmentation


def preprocessing():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def augmentation(c, local=False):
    c = resolve_augmentation(c)
    transforms = [
        T.RandomResizedCrop(c['local_size'] if local else c['global_size'],
                            scale=c['local_scale'] if local else c['global_scale'],
                            ratio=c['crop_ratio'], antialias=True),
        T.RandomHorizontalFlip(c['horizontal_flip']),
        T.RandomVerticalFlip(c['vertical_flip']),
        T.RandomApply([T.ColorJitter(c['jitter'], c['jitter_contrast'],
                                    c['jitter_saturation'], c['jitter_hue'])],
                      p=c['jitter_probability']),
        T.RandomGrayscale(c['grayscale_probability'])]
    # Do not insert disabled transforms: even p=0 wrappers consume RNG.
    if c['blur_probability']:
        transforms.append(T.RandomApply([T.GaussianBlur(c['blur_kernel'], c['blur_sigma'])],
                                        p=c['blur_probability']))
    if c['solarization_probability']:
        transforms.append(T.RandomSolarize(c['solarization_threshold'],
                                           p=c['solarization_probability']))
    return T.Compose(transforms + [preprocessing()])


class Views(Dataset):
    def __init__(self, path, config):
        self.images = load_training_data(path).select_columns(['image'])
        self.config = config
        self.global_transform = augmentation(config)
        self.local_transform = augmentation(config, True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]['image'].convert('RGB')
        return {'global': [self.global_transform(image) for _ in range(self.config['global_views'])],
                'local': [self.local_transform(image) for _ in range(self.config['local_views'])]}
