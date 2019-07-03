
import torchvision.models as models

def build_model(cfg):
    model = models.resnet50(
        pretrained=False,
        num_classes=cfg['dataset']['num_class']
    )
    return model