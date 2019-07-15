import json

from blindness.dataset import build_dataset
from blindness.transforms import build_transforms


config_path = './blindness/configs/base.json'

with open(config_path, 'r') as f:
    cfg = json.loads(f.read())

train_transform  = build_transforms(cfg, split='train')
train_data = build_dataset(cfg, train_transform, split='train')

valid_transform  = build_transforms(cfg, split='valid')
valida_data = build_dataset(cfg, valid_transform, split='valid')

print(len(train_data.dataset))
print(len(valida_data.dataset))
print(len(train_data.dataset)+len(valida_data.dataset))

for i in range(10):
    img, target = train_data.dataset[i]
    print(img.shape, target)