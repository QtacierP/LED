from led.models.unet import UNet2DGenerator
from omegaconf import OmegaConf


def build_model(config):
    if config.model_name == 'unet':
        model_network = UNet2DGenerator.from_config(OmegaConf.to_object(config.model))
        model_class = UNet2DGenerator
    else:
        raise NotImplementedError(f"Model {config.model_name} is not implemented")
    return model_network, model_class
