import json
import time
import torch
import argparse

import pandas as pd
import numpy as np

from torch import cuda
from pathlib import Path
from shutil import copyfile
from torch import optim, nn
from tqdm import tqdm, trange
from sklearn.metrics import cohen_kappa_score

from .configs import dataset_map
from .models import build_model
from .transforms import build_transforms, build_tta_transforms
from .dataset import build_dataset
from .utils import load_checkpoint, save_checkpoint, ON_KAGGLE
from .optimizer import build_optimizer, build_scheduler


if not ON_KAGGLE:
    from torch.utils.tensorboard import SummaryWriter

def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train','valid', 'predict', 'submit'])
    arg('--config_path', type=str, default='blindness/configs/base.json')
    arg('--tta', type=int, default=0)
    arg('--model_path', type=str)
    arg('--predictions', nargs='+')
    arg('--output_path', type=str, default='submission.csv')
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
    elif args.mode == 'valid':
        run_valid(cfg, args.model_path, args.num_tta)
    elif args.mode == 'predict':
        predict(cfg, args.model_path, args.num_tta)
    elif args.mode == 'submit':
        submit(args.predictions, args.output_path)
    else:
        raise NotImplementedError

def train(cfg):
    if cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    train_transform  = build_transforms(cfg, split='train')
    valid_transform  = build_transforms(cfg, split='valid')

    train_data = build_dataset(cfg, train_transform, split='train')
    valid_data = build_dataset(cfg, valid_transform, split='valid')
    
    model = build_model(cfg, device)

    since = time.time()
    lr = cfg['train_param']['lr']
    num_epochs = cfg['train_param']['epoch']
    grad_clip_step = cfg['train_param']['grad_clip_step']
    grad_clip = cfg['train_param']['grad_clip']

    output_dir = Path('output', cfg['name'])
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / 'model.pt'
    best_model_path = output_dir / 'best_model.pt'
    start_epoch = 0
    best_valid_score = 0
    best_valid_loss = float('inf')
    loss_list = []

    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=4))
    
    if not ON_KAGGLE:
        writer = SummaryWriter(log_dir=output_dir / 'tensorboard')

    if model_path.exists(): # modelpath가 있을 경우 load
        start_epoch, best_valid_score, best_valid_loss, lr = load_checkpoint(model, model_path)
        start_epoch += 1

    optimizer = build_optimizer(cfg, model, lr)
    scheduler = build_scheduler(cfg, optimizer)

    step = 0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0

        for image, target, _ in tqdm(train_data,  desc='Train'):            
            loss = model(image, target)
            batch_size = image.size(0)
            (batch_size * loss).backward()
            if step > grad_clip_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            running_loss += loss.item() * image.size(0)


        epoch_loss = running_loss / len(train_data)
        loss_list.append(epoch_loss)
        print('Train Loss: {:.4f}'.format(epoch_loss))
    
        valid_loss, valid_score = validate(model, valid_data, cfg) # Validation
        save_checkpoint(model, model_path, epoch, best_valid_score, best_valid_loss, lr) # Save checkpoint
        # Update best model, loss, score
        if valid_loss < best_valid_loss: best_valid_loss = valid_loss
        if valid_score > best_valid_score: 
            best_valid_score = valid_score
            copyfile(model_path, best_model_path) # copy modelfile to best model

        # write to tensorboard
        if not ON_KAGGLE:
            if scheduler is not None: lr = scheduler.get_lr()[-1]
            writer.add_scalar('loss', running_loss, global_step=epoch)
            writer.add_scalar('lr', lr, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('valid_score', valid_score, global_step=epoch)
            writer.add_scalar('best_valid_score', best_valid_score, global_step=epoch)
            writer.add_scalar('best_valid_loss', best_valid_loss, global_step=epoch)
        if scheduler is not None:
            scheduler.step()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return None

def validate(model, valid_data, cfg):
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []

    with torch.no_grad():
        for i, item in tqdm(enumerate(valid_data), desc='Valid'):
            image, target, _ = item
            batch_size = image.size(0)
            outputs, loss = model(image, target, validate=True)
            all_losses.append(loss.detach().cpu().numpy())
            start_index = i*batch_size
            end_index = len(valid_data) if (i+1)*batch_size > len(valid_data) else (i+1)*batch_size

            all_losses.append(loss)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(target.numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        valid_loss = sum([loss.item() for loss in all_losses])/len(all_losses)
        valid_score = cohen_kappa_score(all_targets, np.argmax(all_predictions, axis=1), weights='quadratic')
        print('Validation Loss: {:.4f}'.format(valid_loss))
        print('val_score: {:.4f}'.format(valid_score))

    return valid_loss, valid_score

def run_valid(cfg, model_path, num_tta):
    if cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    valid_transform  = build_transforms(cfg, split='valid')
    valid_data = build_dataset(cfg, valid_transform, split='valid')

    output_dir = Path('output', cfg['name'])
    output_dir.mkdir(exist_ok=True, parents=True)
    best_model_path = output_dir / 'best_model.pt'
    
    model = build_model(cfg, device)

    if model_path is not None:
        load_checkpoint(model, model_path, False)
    elif best_model_path.exists(): # best modelpath가 있을 경우 load
        load_checkpoint(model, best_model_path, False)

    valid_loss, valid_score = validate(model, valid_data, cfg)
    print('Validation Loss: {:.4f}'.format(valid_loss))
    print('val_score: {:.4f}'.format(valid_score))

def predict(cfg, model_path, num_tta):
    cfg['model']['pretrained'] = False
    if cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    test_transform  = build_transforms(cfg, split='test')
    test_data = build_dataset(cfg, test_transform, split='test')
    
    model = build_model(cfg, device)
    output_dir = Path('output', cfg['name'])
    output_dir.mkdir(exist_ok=True, parents=True)
    best_model_path = output_dir / 'best_model.pt'
    pred_path = output_dir / 'prediction.pt'
    num_class = cfg['dataset']['num_class']

    if model_path is not None:
        load_checkpoint(model, model_path, False)
    elif best_model_path.exists(): # best modelpath가 있을 경우 load
        load_checkpoint(model, best_model_path, False)
    

    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for image, ids in tqdm(test_data, desc='Predict'):
            outputs = model(image)
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(num_class)))
    df = df.groupby(level=0).mean()
    df.to_hdf(pred_path, 'prob', index_label='id_code')
    print('Saved predictions to {}'.format(pred_path))

def submit(predictions, output):
    sample_submission = pd.read_csv(
        dataset_map['submission'],
        index_col='id_code'
    )
    dfs = []
    for prediction in predictions:
        df = pd.read_hdf(prediction, index_col='id_code')
        df = df.reindex(sample_submission.index)
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.groupby(level=0).mean() # get mean of predictions
    df = df.apply(get_classes, axis=1)
    df.name = 'diagnosis'
    df.to_csv(output, header=True)


def get_classes(item):
    return item.idxmax()


if __name__ == '__main__':
    main()