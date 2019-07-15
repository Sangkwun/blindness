
import torch

import torchvision.models as models

from torch import nn
from torch.nn import functional as F


model_map = {
    "resnet50": models.resnet50
}

class Model(nn.Module):
    def __init__(self, cfg, device):
        super(Model, self).__init__()

        self.device = device
        self.mode = cfg['dataset']['method']
        weights_path = cfg['model']['weight_path']
        
        if self.mode == 'classification':
            out_feature=cfg['dataset']['num_class']
            self.criterion = nn.CrossEntropyLoss()
        elif self.mode == 'regression':
            out_feature=1
            self.criterion = nn.MSELoss()
        else: 
            raise ValueError

        args = (
            cfg['model']['pretrained'],
            cfg['dataset']['num_class']
        )
        self.backbone = model_map[cfg['model']['name']](*args)

        if weights_path is not None:
            self.backbone.load_state_dict(torch.load(weights_path))

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=out_feature, bias=True),
        )

    def loss(self, pred, label):
        if self.mode == 'regression':
            label = label.float()
        loss = self.criterion(pred, label)
        return loss

    def forward(self, input_img, target=None, validate=False):
        input_img = input_img.to(self.device)
        if target is not None:
            target = target.to(self.device)
        pred = self.backbone(input_img)

        if self.training or validate: # training time return loss
            loss = self.loss(pred, target)

        if self.training:
            return loss
        elif validate:
            if self.mode == 'classification':
                pred =  F.softmax(pred)
            return pred, loss
        else:
            if self.mode == 'classification':
                pred =  F.softmax(pred)
            return pred


def build_model(cfg, device):
    model = Model(cfg, device)
    model.to(device)
    return model