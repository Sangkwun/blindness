import json
import argparse
import time

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm, trange
from torch import optim
from torch import nn
from sklearn.metrics import cohen_kappa_score

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
    device = torch.device("cuda:0")

    train_transform  = build_transforms(cfg, split='train')
    valid_transform  = build_transforms(cfg, split='valid')

    train_data = build_dataset(cfg, train_transform, split='train')
    valid_data = build_dataset(cfg, valid_transform, split='valid')
    model = build_model(cfg, device)
    since = time.time()
    
    # cfg에 따라서 optimizer를 바꿔 적용
    optimizer = optim.Adam(model.parameters(), lr=cfg['train_param']['lr'])
    
    loss_list = []
    num_epochs = cfg['train_param']['epoch']
    # -----train------
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        counter = 0

        for image, target in tqdm(train_data):            
            loss = model(image, target)
            batch_size = image.size(0)
            (batch_size * loss).backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * image.size(0)
            counter += 1
        epoch_loss = running_loss / len(train_data)
        loss_list.append(epoch_loss)
        print('Train Loss: {:.4f}'.format(epoch_loss))

    
        validate(model, valid_data)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return None

def validate(model, valid_data):
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []

    with torch.no_grad():
        for i, item in tqdm(enumerate(valid_data)):
            image, target = item
            batch_size = image.size(0)
            outputs, loss = model(image, target, validate=True)
            all_losses.append(loss.detach().cpu().numpy())
            start_index = i*batch_size
            end_index = len(valid_data) if (i+1)*batch_size > len(valid_data) else (i+1)*batch_size

            predictions = torch.sigmoid(outputs)

            all_losses.append(loss)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        valid_loss = sum([loss.item() for loss in all_losses])
        valid_score = cohen_kappa_score(all_targets, np.argmax(all_predictions, axis=1), weights='quadratic')
        print('Validation Loss: {:.4f}'.format(valid_loss))
        print('val_score: {:.4f}'.format(valid_score))

    return valid_loss, valid_score

def predict(cfg):

    return None

if __name__ == '__main__':
    main()