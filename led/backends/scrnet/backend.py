from led.backends.base_backend import BaseBackend
from led.backends.scrnet.network import SCRNetwork
from bunch import Bunch
import torch
import numpy as np
import cv2
from scipy import ndimage
import torch.utils.model_zoo as tm

class SCRNetBackend(BaseBackend):
    def __init__(self):
        super().__init__()
    
    def _set_default_hyperparameters(self):
        hyperparameters = {
                'model_name': 'scr-net',
                'model_path': 'pretrained_weights/scrnet.pth',
                'image_size': 256,
                'image_mean': 0.5,
                'image_std': 0.5,
                'n_downs': 8,
                'input_nc': 3,
                'output_nc': 3,
                'n_filters': 64,
                'filter_width': 27,
                'nsig': 9,
                'ratio': 4, 
                'sub_low_ratio': 1.0,
                'use_dropout': False}
        self.hyperparameters = Bunch(hyperparameters)
        
                
    def _build_model(self):
        self.model = SCRNetwork(self.hyperparameters)
        try:
            self.model.netG.load_state_dict(torch.load(self.hyperparameters.model_path, map_location='cpu'))
        except:
            self.model.netG.load_state_dict(tm.load_url('https://github.com/QtacierP/LED/releases/download/weights/scrnet.pth', model_dir='pretrained_weights', map_location='cpu'))
    
    def _preprocess(self, input, **kwargs):
        # resize input to defalue size
        if self.hyperparameters.image_size != input.shape[2]:
            input = torch.nn.functional.interpolate(input, 
            size=self.hyperparameters.image_size, mode='bilinear', align_corners=True)
        # unnorm   
        image = input * self.hyperparameters.image_std + self.hyperparameters.image_mean
        image = image.permute(0, 2, 3, 1)
        # convert to cv2 and extract mask
        image = image.cpu().numpy()
        mask = np.zeros((image.shape[0], image.shape[1], image.shape[2], 1))
        for i in range(image.shape[0]):
            mask[i, :, :, 0] = self._get_mask(image[i, :, :, :])
        mask = mask.transpose(0, 3, 1, 2)
        mask = torch.from_numpy(mask).to(input.device).float()
        # resize input to defalue size
        return input, mask
    

    def _get_mask(self, img):
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gra_normalize = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return ndimage.binary_opening(gra_normalize > 10, structure=np.ones((8, 8)))
    
    def _forward(self, input, mask):
        return self.model(input, mask)
    
    def _post_precess(self, image):
        if self.image_size != image.shape[2]:
            image = torch.nn.functional.interpolate(image, size=self.image_size, mode='bilinear')
        return image

    def __call__(self, input):
        self.image_size = input.shape[2]
        input, mask = self._preprocess(input)
        return self._post_precess(self._forward(input, mask))


    
