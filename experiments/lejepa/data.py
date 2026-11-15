"""Image-only dataset: labels are removed before examples enter the loader."""
from torch.utils.data import Dataset
from torchvision import transforms as T
from training_data import load_training_data
from export_onnx import MEAN, STD


def preprocessing():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def augmentation(c, local=False):
    return T.Compose([
        T.RandomResizedCrop(c['local_size'] if local else 96,
                            scale=c['local_scale'] if local else c['global_scale'],
                            ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(c['horizontal_flip']),
        T.RandomVerticalFlip(c['vertical_flip']),
        T.RandomApply([T.ColorJitter(c['jitter'], c['jitter'], c['jitter'], 0.02)],
                      p=c['jitter_probability']),
        T.RandomGrayscale(c['grayscale_probability']), preprocessing()])


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
