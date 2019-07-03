
import json
import argparse

import pandas as pd
import numpy as np

from .configs import dataset_map
from .models import build_model
from .transforms import build_transforms
from .dataset import build_dataset

def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'predict', 'submit'])
    arg('--config_path', type=str, default='blindness/configs/base.json')

    # for run
    arg('--run_name', default='stage1', type=str)
    arg('--model_configs', nargs='+')

    # for split_fold
    arg('--n_fold', type=int, default= 5)

    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    with open(args.config_path, 'r') as f:
        cfg = json.loads(f.read())

    if args.mode == 'train':
        train(cfg)
    elif args.mode == 'predict':
        predict(cfg)
    elif args.mode == 'submit':
        raise NotImplementedError
    else:
        raise NotImplementedError


def train(cfg):
    train_transform  = build_transforms(cfg, split='train')
    valid_transform  = build_transforms(cfg, split='valid')

    train_data = build_dataset(cfg, train_transform, split='train')
    valid_data = build_dataset(cfg, valid_transform, split='valid')
    print(train_data)

    model = build_model(cfg)
    print(model)
    return None

def predict(cfg):

    return None

if __name__ == '__main__':
    main()