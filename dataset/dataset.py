import torchaudio
import torch
from torch.utils.data import Dataset
import os


class AudioData(Dataset):

    def __init__(self, audio_path, sample_rate, sample_len):
        super().__init__()
        self.audio_path = audio_path
        self.target_sample_rate = sample_rate
        self.sample_len = sample_len
        self.audio_files = os.listdir(audio_path)
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        audio_sample_path = self.get_audio_path(index)
        if not os.path.exists(audio_sample_path):
            raise FileNotFoundError(f'Audio file {audio_sample_path} not found.')
        try:
            signal, sr = torchaudio.load(audio_sample_path)
        except Exception as e:
            raise RuntimeError(f'Could not open {audio_sample_path}. {e}')
        
        signal = self.resample(signal, sr)
        signal = self.to_mono(signal)
        signal = self.padding(signal)
        signal = self.truncate(signal)
        return signal

    def get_audio_path(self, index):
        filename = self.audio_files[index]
        return os.path.join(self.audio_path, filename)

    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal 

    def to_mono(self, signal):
        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def padding(self, signal):
        len_signal = signal.shape[1]
        if len_signal < self.sample_len:
            n_missing = self.sample_len - len_signal
            padding = (0, n_missing)
            signal = torch.nn.functional.pad(signal, padding)
        return signal
    
    def truncate(self, signal):
        if signal.shape[1] > self.sample_len:
            signal = signal[:, :self.sample_len]
        return signal

        
if __name__ == "__main__":
    
    dataset = AudioData(
        audio_path='/Users/inigoparra/Desktop/ESC-50-master/audio',
        sample_rate=8000,
        sample_len=40000
    )