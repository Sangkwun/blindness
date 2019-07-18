import random

from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale

class RandomRotate(object):
    def __init__(self):
        self.degrees = [0, 90, 180, 270]
    def __call__(self, img):
        degree = random.choice(self.degrees)
        return img.rotate(degree)


def build_transforms(cfg, split='train'):
    is_train = split == 'train'
    input_cfg = cfg['input']
    width = input_cfg['width']
    height = input_cfg['height']

    transforms = []
    for transform in input_cfg['transforms']:
        if transform == 'random_resized_crop':
            scale = (0.8, 1.2) if is_train else (1.0, 1.0)
            ratio = (1.0, 1.0) if is_train else (1.0, 1.0)
            transforms.append(
                RandomResizedCrop(
                    (width, height),
                    scale=scale,
                    ratio=ratio,
                )
            )
        elif transform == 'random_rotate':
            transforms.append(RandomRotate())
        elif transform == 'random_vertical_flip':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomVerticalFlip(p))
        elif transform == 'random_horizontal_flip':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomHorizontalFlip(p))
        elif transform == 'random_color_jitter':
            brightness = 0.1 if is_train else 0.0
            contrast = 0.1 if is_train else 0.0
            transforms.append(ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0,
                hue=0,
            ))
        elif transform == 'random_grayscale':
            p = 0.5 if is_train else 0.25
            transforms.append(RandomGrayscale(p))
        else:
            print(transform)
            raise NotImplementedError

    transforms += [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return Compose(transforms)