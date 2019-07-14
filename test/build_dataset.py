import json

from blindness.dataset import build_dataset
from blindness.transforms import build_transforms


config_path = './blindness/configs/base.json'

with open(config_path, 'r') as f:
    cfg = json.loads(f.read())

train_transform  = build_transforms(cfg, split='train')
train_data = build_dataset(cfg, train_transform, split='train')

print(len(train_data.dataset))

for i in range(10):
    img, target = train_data.dataset[i]
    print(img.shape, target)