import yaml
import argparse
import os
import sys
# add current path to workspace
sys.path.append(os.getcwd())
from led.trainers.led_trainer import Trainer
from bunch import Bunch
import os
import torch
from omegaconf import OmegaConf, listconfig



# make argument parser
parser = argparse.ArgumentParser(description='LED')
parser.add_argument('--config', type=str, default='./configs/train_led.yaml', help='config file')
parser.add_argument('--resume', type=str, default='', help='config file')


# parse arguments
args = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(args.config)
    if args.resume != '':
        config.resume = args.resume
    else:
        config.resume = None
    Trainer(config).train()




