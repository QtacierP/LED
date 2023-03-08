from datasets import load_dataset,  concatenate_datasets
import datasets
from torchvision import transforms
from glob import glob
import os
from pytorch_lightning.utilities.seed import seed_everything
import random
import torch
import numpy as np
from PIL import Image


def build_dataset(good_image_dir, bad_image_dir, config,  degraded_image_dir=None,  train=True):
    good_images = glob(os.path.join(good_image_dir, '*'))
    bad_images = glob(os.path.join(bad_image_dir, '*'))
    if degraded_image_dir is not None:
        degraded_images = glob(os.path.join(degraded_image_dir, '*'))
        good_dataset = datasets.Dataset.from_dict({"image": good_images,  "degraded_image": degraded_images}, features=datasets.Features({"image": datasets.Image(), "degraded_image": datasets.Image()}))
        good_dataset.cleanup_cache_files()
    
    else:
        good_dataset = datasets.Dataset.from_dict({"image": good_images}, features=datasets.Features({"image": datasets.Image()}))
    bad_dataset = datasets.Dataset.from_dict({"image": bad_images}, features=datasets.Features({"image": datasets.Image()}))
    if train:
        preprocess = transforms.Compose(
                    [
                        transforms.Resize((config.data.image_size, config.data.image_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )
    else:
        preprocess = transforms.Compose(
                    [
                        transforms.Resize((config.data.image_size, config.data.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )
        
    def transform(examples):
        for idx in range(len(examples['image'])):
            # fix the seed for each image
            seed = random.randint(0, 2**32)
            for key in examples.keys():
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                examples[key][idx] = preprocess(examples[key][idx].convert('RGB'))
        return examples
    good_dataset.set_transform(transform)
    bad_dataset.set_transform(transform)
    return good_dataset, bad_dataset



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform):
        self.images = sorted(glob(os.path.join(dir, '*')))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.images[index]).convert('RGB'))
        return image, self.images[index]



        

    
    
    
