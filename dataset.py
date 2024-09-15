import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class AudioData(Dataset):
    def __init__(self, audio_path, audio_labels, transform):
        super().__init__()
        self.audio_path = audio_path
        self.audio_labels = pd.read_csv(audio_labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.audio_path)
    
    def __getitem__(self, index):
        pass
        