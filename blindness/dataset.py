import torch
import pandas as pd
from glob import glob
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .configs import dataset_map, diabetic_retinopathy_map


class BlindDataset(Dataset):
    """
    fold가 구분되어 train 또는 test로 split된 dataframe을 입력으로 받음.
    root: dataset root dir
    df: label을 포함한 dataframe
    num_class : label의 갯수
    output : label의 형태로 classification일 경우 one-hot 형태, regression일 경우 0, 1, 2, 3, 4 형태
    """
    def __init__(self, image_dir, df, transforms, num_class, is_test):
        super().__init__()
        self.is_test = is_test
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        self.num_class = num_class

    def __len__(self):
        return len(self.df)

    def get_img(self, index):
        if 'id_code' not in self.df.keys():
            img_name = self.df.iloc[index]['image']
            img_path = self.image_dir + '/' + img_name + '.jpeg'
        else:
            img_name = self.df.iloc[index]['id_code']
            img_path = self.image_dir + '/' + img_name + '.png'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        return image

    def __getitem__(self, index):
        item = self.df.iloc[index]
        image = self.get_img(index)
        target = None
        if self.transforms is not None:
            image = self.transforms(image)
        if 'diagnosis' not in self.df.keys():
            if not self.is_test:
                target = torch.tensor(item['level'])
            ids = item['image']
        else:
            if not self.is_test:
                target = torch.tensor(item['diagnosis'])
            ids = item['id_code']
        return image, target, ids


def build_dataset(cfg, transforms,  split='train'):
    assert split in ['train', 'valid', 'test']
    dataset_config = cfg['dataset']

    num_class = dataset_config['num_class']
    fold = dataset_config['fold']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']
    method = dataset_config['method']
    is_test = split == 'test'

    if split == 'test':
        df = pd.read_csv(dataset_map['test'])
        image_dir = dataset_map['test_images']
    else:
        df = pd.read_csv(dataset_map['fold'])
        image_dir = dataset_map['train_images']
    
    if split == 'train':
        df = df[df['fold'] != fold]
    elif split == 'valid':
        df = df[df['fold'] == fold]

    dataset = BlindDataset(
        image_dir=image_dir,
        df=df,
        transforms=transforms,
        num_class=num_class,
        is_test=is_test
    )
    if split == 'train' and dataset_config['use_diabetic_retinopathy']:
        diabetic_df = pd.read_csv(diabetic_retinopathy_map['train'], index_col='Unnamed: 0')
        del diabetic_df['Unnamed: 0.1']
        diabetic_dataset = BlindDataset(
            image_dir=diabetic_retinopathy_map['train_images'],
            df=diabetic_df,
            transforms=transforms,
            num_class=num_class,
            is_test=is_test
        )

        dataset = ConcatDataset([dataset, diabetic_dataset])
    data_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return data_loader