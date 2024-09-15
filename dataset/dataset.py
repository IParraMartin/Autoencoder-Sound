import torchaudio
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
        audio_sample = self.get_audio_path(index)

        if not os.path.exists(audio_sample):
            raise FileNotFoundError(f'Audio file {audio_sample} not found.')
        
        try:
            signal, sr = torchaudio.load(audio_sample)
        except Exception as e:
            raise RuntimeError(f'Could not open {audio_sample}. {e}')
        
        signal = self.get_audio_path(signal)
        signal = self.resample(signal, sr)


    def get_audio_path(self, index):
        return os.path.join(self.audio_path, index)

    def resample(self, sr, signal):
        pass

    def to_mono(self, signal):
        pass
        