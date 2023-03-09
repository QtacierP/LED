# Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement

The official implementation of the paper [Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement](https://arxiv.org/pdf/2303.04603.pdf)

<p align="center">
<img src="https://img.shields.io/github/license/QtacierP/LED" />
 <img src="https://img.shields.io/github/issues/QtacierP/LED" />
 <img src="https://img.shields.io/github/forks/QtacierP/LED" />
 <img src="https://img.shields.io/github/stars/QtacierP/LED" />
</p>

<img src="./docs/led.gif"  width="300" height="300">

## Highlights
- ### Continuous and reliable enhancement
![image info](./docs/Continuous.png)

- ### Flexable and ffectively integrated into any SOTA
![image info](./docs/OC.png)

![image info](./docs/vessels.png)

- ### Robust for OOD low-quality images
![image info](./docs/jpeg_loss.png)

![image info](./docs/mf_loss.png)

- ### SOTA performance
![image info](./docs/performance.png)

Start LED with few lines

```python
from led.pipelines.led_pipeline import LEDPipeline
led = LEDPipeline()
led.cuda()
led_enhancement = led('./doc/example.jpeg')[0]
```

Furthermore, you can combine LED with any existing SOTA methods as external backend. Current supported backends include:
- [I-SECRET](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_9), MICCAI 2021
- [PCE-Net](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_49), MICCAI 2022
- [SCRNet](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_47), MICCAI 2022
- [ArcNet](https://ieeexplore.ieee.org/document/9698071/), TMI 2022

Try I-SECRET backend with only one line
```python
led = LEDPipeline(backend='I-SECRET', num_cond_steps=200)
```

For more details, please read [example.ipynb](example.ipynb). Please feel free to pull your proposed fundus enhancement methods as backend.


## Catalog
- [ ] Training guidance
- [x] Support for ArcNet and SCRNet
- [ ] Add related codes for data-driven degradation
- [x] Inference pipeline

## Train
For training your own LED, you need to update few lines in configs/train_led.yaml
```yaml
    train_good_image_dir: # update to training hq images directrory
    train_bad_image_dir: # update to training lq images directrory
    train_degraded_image_dir: # update to training degraded images directrory
    val_good_image_dir:  # update to validation hq images directrory
    val_bad_image_dir: # update to validation lq images directrory
```
Please note that ``train_degraded_image_dir`` should contain degraded high-qualty images by any data-driven methods. We will inculde related codes in our future workspace. However, you can consider using some existing repos instead, like [CUT](https://github.com/taesungp/contrastive-unpaired-translation) or [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

To train LED, simply  run
```bash
accelerate launch  --mixed_precision fp16 --gpu_ids 0 --num_processes 1 script/train.py 
```
More GPUs will take significant performance improvement.


## Acknowledgement 
Thanks for [PCENet](https://github.com/HeverLaw/PCENet-Image-Enhancement), [ArcNet](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement) and [SCRNet](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement) for sharing their powerful pre-trained weights! Thansk for [diffusers](https://github.com/huggingface/diffusers) for sharing codes.

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{cheng2023learning,
  title={Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement},
  author={Cheng, Pujin and Lin, Li and Huang, Yijin and He, Huaqing and Luo, Wenhan and Tang, Xiaoying},
  journal={arXiv preprint arXiv:2303.04603},
  year={2023}
} 
```
## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.