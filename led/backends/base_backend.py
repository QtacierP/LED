import torch
from abc import ABC



class BaseBackend(ABC):
    def __init__(self):
        self._set_default_hyperparameters()
        self._build_model()

    def _set_default_hyperparameters(self):
        pass

    def cuda(self):
        self.model.cuda()
    
    def cpu(self):
        self.model.cpu()

    def _preprocess(self, input, **kwargs):
        return input

    def _forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def __call__(self, input, **kwargs):
        input = self._preprocess(input, **kwargs)
        return self._forward(input, **kwargs)
    
    def to(self, device):
        self.model.to(device)

    


        