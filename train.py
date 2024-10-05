import torch
from dataset.dataset import AudioData
from src.ae import Autoencoder
import argparse
import yaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.parse_args('--configfile', type=str, help='')
    args = parser.parse_args()

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model = Autoencoder(
        in_dims=16000,
        h_dims=10,
        is_image=False
    )
    
