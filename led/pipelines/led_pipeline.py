# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import  List, Optional, Union
import numpy as np
import torch
import PIL
from diffusers.utils import logging, randn_tensor
from diffusers import DDPMScheduler
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from PIL import Image
from led.models.unet import UNet2DGenerator, _default_config
from led.backends.isecret import ISECRETBackend
from led.backends.pcenet import PCENetBackend
from omegaconf import OmegaConf
import logging


logger = logging.getLogger(__name__)







class LEDPipeline(DiffusionPipeline):
    r"""
    Pipeline for Learning Enhancement from learning Degradation (LED).
    """
    def __init__(self, unet=None, scheduler=None, base_pipeline='ddim', backend=None, num_cond_steps=800):
        super().__init__()
        # make sure scheduler can always be converted to DDIM (basically)
        if scheduler is None:
            scheduler = self._set_default_scheduler()
        if unet is None:
            self.unet = self._build_unet()
        if base_pipeline == 'ddim':
            from diffusers.schedulers import DDIMScheduler
            self.scheduler = DDIMScheduler.from_config(scheduler.config)
        else:
            raise NotImplementedError(f'base_pipeline {base_pipeline} not implemented')
        self.backend = self._build_backend(backend)
        self.register_modules(unet=self.unet, scheduler=self.scheduler, backend=self.backend)
        self.num_cond_steps = num_cond_steps


    def _set_default_scheduler(self):
        logger.info('Using default scheduler.')
        return DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule='linear',
                prediction_type='epsilon',
            )
    
    def _build_unet(self):
        unet = UNet2DGenerator.from_config(OmegaConf.to_object(_default_config)).eval()
        # load from pretrained path or url
        try:
            state_dict = torch.load('pretrained_weights/led.bin', map_location='cpu')
        except:
            raise NotImplementedError('The pretrained weights url are not available yet.')
        unet.load_state_dict(state_dict)
        logger.info('Loading pretrained weights for unet.')
        return unet

    def _build_backend(self, backend):
        self.backend = None
        if backend is None:
            logger.info('Using LED directly.')
        elif backend == 'i-secret' or backend == 'I-SECRET':
            self.backend = ISECRETBackend().to(self.unet.device)
            logger.info('Using I-SECRET backend.')
        elif backend == 'pcenet' or backend == 'PCE-Net':
            self.backend = PCENetBackend().to(self.unet.device)
            logger.info('Using PCE-Net backend.')
        else:
            raise NotImplementedError(f'Backend {backend} not implemented.')
        

    def cuda(self):
        self.unet.cuda()
    
    def cpu(self):
        self.unet.cpu()

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        # unnormalize with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
        images = images * 0.5 + 0.5
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    
    def _pre_process(self, image):
        if isinstance(image, torch.Tensor):
            image = self._pre_process_for_tensor(image)
        elif isinstance(image, PIL.Image.Image):
            image = self._pre_process_for_pil(image)
        elif isinstance(image, np.ndarray):
            image = self._pre_process_for_numpy(image)
        elif isinstance(image, str):
            image = self._pre_process_for_str(image)
        elif isinstance(image, list):
            if isinstance(image[0], PIL.Image.Image):
                image = torch.cat([self._pre_process_for_pil(_image) for _image in image], dim=0)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat([self._pre_process_for_tensor(_image) for _image in image], dim=0)
            elif isinstance(image[0], np.ndarray):
                image = torch.cat([self._pre_process_for_numpy(_image) for _image in image], dim=0)
            elif isinstance(image[0], str):
                image = torch.cat([self._pre_process_for_str(_image) for _image in image], dim=0)
            else:
                raise ValueError(f"Invalid input type {type(image[0])}")
        else:
            raise ValueError(f"Invalid input type {type(image)}")
        return image


    def _pre_process_for_pil(self, image):
        image = np.array(image.resize((512, 512)))
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = self._normalize(image).unsqueeze(0)
        return image

    def _pre_process_for_tensor(self, image):
        image = self._normalize(image)
        if image.dim() == 3:
            image = image[None, ...]
        return image
    
    def _pre_process_for_numpy(self, image):
        image = self._normalize(image)
        image = torch.from_numpy(image)
        if image.dim() == 3:
            image = image[None, ...]
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        return image.float()
    
    def _pre_process_for_str(self, image):
        image = Image.open(image)
        image = self._pre_process_for_pil(image)
        return image

    def _normalize(self, image):
        if image.max() > 1:
            return (image - 127.5) / 127.5
        else:
            return (image - 0.5) / 0.5

    @torch.no_grad()
    def __call__(
        self,
        cond_image: Union[torch.Tensor, PIL.Image.Image, List[Union[torch.Tensor, PIL.Image.Image]]],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "numpy",
        output_max_val: int=255,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            cond_image (`string`, `np.array`, `PIL.Image`, `torch.Tensor`):
                 one or list of low-quality images of interest.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            output_max_val (`int`, *optional*, defaults to 255):
                The maximum value of the output image. If `output_type` is `"pil"`, this is ignored.
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # preprocess for input
        cond_image = self._pre_process(cond_image).to(self.unet.device)
        if self.backend is not None:
            cond_image = self.backend(cond_image)
        image = randn_tensor(cond_image.shape, generator=generator, device=self.unet.device, dtype=cond_image.dtype)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        max_T = self.num_cond_steps
        timesteps = torch.ones(cond_image.shape[0], device=self.unet.device, dtype=torch.long) * max_T
        # Add noise to the clean images according to the noise magnitude at each timestep
        image = self.scheduler.add_noise(cond_image, image, timesteps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if t > max_T:
                continue
            inputs = torch.cat([image, cond_image], dim=1)
            model_output = self.unet(inputs, t).sample
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            return self.numpy_to_pil(image)
        image = (image / 2 + 0.5).clip(0, 1)
        if output_max_val == 255:
            image = (image * 255).astype(np.uint8)
        if output_type == 'dict':
            return ImagePipelineOutput(images=image)
        return image


