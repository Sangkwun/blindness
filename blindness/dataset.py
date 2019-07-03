import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from .configs import dataset_map

class BlindDataset(Dataset):
    """
    fold가 구분되어 train 또는 test로 split된 dataframe을 입력으로 받음.
    root: dataset root dir
    df: label을 포함한 dataframe
    num_class : label의 갯수
    """
    def __init__(self, image_dir, df, transforms, num_class):
        super().__init__()
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def get_img(self, index):
        raise NotImplementedError
        return None

    def __getitem__(self, index):
        item = self.df.iloc[index]
        image = self.get_img(index)
        if self.transforms is not None:
            image = transforms(image)

        target = torch.zeros(self.num_class)
        target[item['diagnosis']] = 1
        return image, target


def build_dataset(cfg, transforms,  split='train'):
    assert split in ['train', 'valid', 'test']
    dataset_config = cfg['dataset']

    num_class = dataset_config['num_class']
    fold = dataset_config['fold']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']

    if split == 'test':
        df = pd.read_csv(dataset_map['test'])
        image_dir = dataset_map['train_images']
    else:
        df = pd.read_csv(dataset_map['fold'])
        image_dir = dataset_map['test_images']
    
    if split == 'train':
        df = df[df['fold'] != fold]
    elif split == 'valid':
        df = df[df['fold'] == fold]

    args = (
        image_dir,
        df,
        transforms,
        num_class
    )
    data_loader = DataLoader(
        BlindDataset(*args),
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # TODO: dataloader로 만들기. concat하기.
    return data_loader