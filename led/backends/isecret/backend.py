from led.backends.base_backend import BaseBackend
from led.backends.isecret.network import ISECRETNetwork
from bunch import Bunch
import torch
import torch.utils.model_zoo as tm

class ISECRETBackend(BaseBackend):
    def __init__(self):
        super().__init__()
    
    def _set_default_hyperparameters(self):
        hyperparameters = {
                'model_name': 'i-secret',
                'model_path': 'pretrained_weights/isecret.pt',
                'image_size': 512,
                'image_mean': 0.5,
                'image_std': 0.5,
                'n_blocks': 9,
                'n_downs': 2,
                'n_filters': 64,
                'input_nc': 3,
                'output_nc': 3,
                'use_dropout': False}
        self.hyperparameters = Bunch(hyperparameters)
        
                
    def _build_model(self):
        self.model = ISECRETNetwork(self.hyperparameters).eval()
        try:
            self.model.load_state_dict(torch.load(self.hyperparameters.model_path, map_location='cpu')['weights'])
        except:
            self.model.load_state_dict(tm.load_url('https://github.com/QtacierP/LED/releases/download/weights/isecret.pt', model_dir='pretrained_weights', map_location='cpu')['weights'])
