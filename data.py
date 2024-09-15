import pandas as pd
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset
import torchaudio
import torch


class AudioData(Dataset):
    
    def __init__(self, annotations_dir, audio_dir, transformation, target_sample_rate, n_samples):
        super().__init__()
        self.annotations = pd.read_csv(annotations_dir)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
    

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        audio_sample_path = self.get_audio_sample_path(index)
        label = self.get_audio_sample_label(index)
        if not os.path.exists(audio_sample_path):
            raise FileNotFoundError(f'Audio file {audio_sample_path} not found.')
        
        try:
            signal, sr = torchaudio.load(audio_sample_path)
        except Exception as e:
            raise RuntimeError(f'Error loading {audio_sample_path}. {e}')

        signal = self.resample_if_necessary(signal, sr)
        signal = self.mixdown_if_necessary(signal)
        signal = self.pad_if_necessary(signal)
        signal = self.truncate_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    def get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2]
    

    def resample_if_necessary(self, signal, sr):
        """
        We are resampling the sound to the sample rate we need
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal


    def mixdown_if_necessary(self, signal):
        """
        We are mixing stereo to mono: (n_channels, samples) => (2, 16000) -> (1, 16000)
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    

    def pad_if_necessary(self, signal):
        """
        Right pad the signal if the samples are not met.
        """
        len_signal = signal.shape[1]
        if len_signal < self.n_samples:
            n_missing = self.n_samples - len_signal
            r_padding = (0, n_missing)
            signal = torch.nn.functional.pad(signal, r_padding)
        return signal
 

    def truncate_if_necessary(self, signal):
        """
        Truncate the signal if it is too long.
        """
        if signal.shape[1] > self.n_samples:
            signal = signal[:, :self.n_samples]
        return signal
    

    @staticmethod
    def plot_example(signal, title: str = None, save: bool = False):
        spectrogram = signal.log2()
        spectrogram = spectrogram.squeeze().detach().numpy()
        plt.figure(figsize=(8, 4))
        plt.imshow(spectrogram, cmap='magma', origin='lower', aspect='auto')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title if title else 'Spectrogram (dB)')
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        if save == True:
            plt.savefig('example.png', dpi=1200)
        
        plt.show()
    

if __name__ == "__main__": 

    annotations = '/Users/inigoparra/Desktop/ESC-50-master/meta/esc50.csv'
    audio_dir = '/Users/inigoparra/Desktop/ESC-50-master/audio'
    sample_rate = 22_050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128 # More fine grained, can be reduced to 64
    )

    usd = AudioData(
        annotations_dir=annotations, 
        audio_dir=audio_dir,
        transformation=mel_spectrogram,
        target_sample_rate=sample_rate,
        n_samples=44_100
    )

    assert str(torchaudio.list_audio_backends()) is not None, 'Try <pip install soundfile> or <pip3 install soundfile>'

    print(f'Total length of the dataset: {len(usd)}')
    signal, label = usd[2]
    print(f'Signal: {signal}\nLabel: {label}')

    usd.plot_example(signal, title=f"Label: {label}")
