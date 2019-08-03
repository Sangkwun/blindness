import os
import torch

ON_KAGGLE = 'KAGGLE_WORKING_DIR' in os.environ

def load_checkpoint(model, path, return_others=True):
    data = torch.load(path)
    model.load_state_dict(data['model'])
    epoch = data['epoch']
    best_valid_score = data['best_valid_score']
    best_valid_loss = data['best_valid_loss']
    lr = data['lr']
    print("Load model from {}".format(path))
    return epoch, best_valid_score, best_valid_loss, lr

def save_checkpoint(model, path, epoch, best_valid_score, best_valid_loss, lr):
    data = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_valid_score": best_valid_score,
        "best_valid_loss": best_valid_loss,
        "lr": lr
    }
    torch.save(data, path)


def mse_decode(preds):
    coef = [0.5, 1.5, 2.5, 3.5]

    for i, pred in enumerate(preds):
        if pred < coef[0]:
            preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            preds[i] = 3
        else:
            preds[i] = 4
    return preds