from torch import nn
import torch
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, cfg):
        self.model = models.resnet50(
            pretrained=False,
            num_classes=cfg['dataset']['num_class']
        )
        cfg
    
    def forward(self, input, target):
        output = self.model(input)
        loss = None
        label = None
        if self.training:
            return loss
        return label


def build_model(cfg, device):
    if cfg['model']['name'] == 'resnet50':
        if cfg['dataset']['method'] == 'classification':
            out_feature=cfg['dataset']['num_class']
        elif cfg['dataset']['method'] == 'regression':
            out_feature=1

        model = models.resnet101(pretrained=False)
        model.load_state_dict(torch.load(cfg['model']['model_path']))

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Sequential(
                                nn.BatchNorm1d(2048, track_running_stats=True),
                                nn.Dropout(p=0.25),
                                nn.Linear(in_features=2048, out_features=2048, bias=True),
                                nn.ReLU(),
                                nn.BatchNorm1d(2048, track_running_stats=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=2048, out_features=out_feature, bias=True),
                                )

        model = model.to(device)

        # model = models.resnet50(
        #     pretrained=False,
        #     num_classes=cfg['dataset']['num_class']
        # )
    
    return model