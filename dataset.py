import torch
from torch.utils.data import Dataset
import os

class AudioData(Dataset):
    def __init__(self, audio_path, sample_rate, sample_len, transform):
        super().__init__()
        self.audio_path = audio_path
        self.target_sample_rate = sample_rate
        self.sample_len = sample_len
        self.transform = transform
        
    def __len__(self):
        return len(self.audio_path)
    
    def __getitem__(self, index):
        pass

    def get_audio_path(self, index)
        pass

    def resample(self, sample, target):
        pass

    def to_mono(self, sample):
        pass
        