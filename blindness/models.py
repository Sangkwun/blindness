from torch import nn
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


def build_model(cfg):
    model = models.resnet50(
        pretrained=False,
        num_classes=cfg['dataset']['num_class']
    )
    return model