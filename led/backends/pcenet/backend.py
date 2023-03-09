from led.backends.base_backend import BaseBackend
from led.backends.pcenet.network import PCENetwork
from bunch import Bunch
import torch
import numpy as np
import cv2
from scipy import ndimage
import torch.utils.model_zoo as tm

class PCENetBackend(BaseBackend):
    def __init__(self):
        super().__init__()
    
    def _set_default_hyperparameters(self):
        hyperparameters = {
                'model_name': 'pce-net',
                'model_path': 'pretrained_weights/pcenet.pth',
                'image_size': 256,
                'image_mean': 0.5,
                'image_std': 0.5,
                'n_blocks': 9,
                'input_nc': 3,
                'output_nc': 3,
                'n_filters': 64,
                'use_dropout': False}
        self.hyperparameters = Bunch(hyperparameters)
        
                
    def _build_model(self):
        self.model = PCENetwork(self.hyperparameters)
        try:
            self.model.netG.load_state_dict(torch.load(self.hyperparameters.model_path, map_location='cpu'))
        except:
            self.model.netG.load_state_dict(tm.load_url('https://github.com/QtacierP/LED/releases/download/weights/pcenet.pth', model_dir='pretrained_weights', map_location='cpu'))
    
    def _preprocess(self, input, **kwargs):
        # unnorm    
        image = input * self.hyperparameters.image_std + self.hyperparameters.image_mean
        image = image.permute(0, 3, 1, 2)
        # convert to cv2 and extract mask
        image = image.numpy()
        mask = np.zeros((image.shape[0], 1, image.shape[2], image.shape[3]))
        for i in range(image.shape[0]):
            mask[i, 0, :, :] = self._get_mask(input[i, :, :, :])
        mask = torch.from_numpy(mask)
        # resize input to defalue size
        if self.hyperparameters.image_size != image.shape[2]:
            input = torch.nn.functional.interpolate(image, size=self.hyperparameters.image_size, mode='bilinear', align_corners=True)
        return input, mask
    


    def _get_mask(img):
        gray = np.array(img.convert('L'))
        gra_normalize = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return ndimage.binary_opening(gra_normalize > 10, structure=np.ones((8, 8)))
    
    def _forward(self, input, mask):
        return self.model(input, mask)
    
    def _post_precess(self, image):
        if self.image_size != image.shape[2]:
            image = torch.nn.functional.interpolate(image, size=self.image_size, mode='bilinear', align_corners=True)
        return image

    def __call__(self, input, mask):
        self.image_size = input.shape[2]
        input, mask = self._preprocess(input)
        return self._forward(input, mask)
    
