import os
import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, target_duration = 1.25):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_samples = int(target_duration * sample_rate)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )

    def load_audio(self, file_path):
        """Load audio file and convert to wav format if necessary"""
        if file_path.endswith('.m4a'):
            audio = AudioSegment.from_file(file_path, format='m4a')
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(1)
            samples = torch.tensor(audio.get_array_of_samples()).float()
        else:
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                samples = resampler(waveform)
            samples = waveform

        audio = samples.mean(dim=0) if samples.dim() > 1 else samples

        # Pad or truncate to fixed length
        if audio.size(0) > self.target_samples:
            audio = audio[:self.target_samples]
        elif audio.size(0) < self.target_samples:
            pad_amount = self.target_samples - audio.size(0)
            audio = torch.nn.functional.pad(audio, (0, pad_amount))

        return audio

    def compute_melspectrogram(self, audio):
        """Compute log-mel spectrogram from audio waveform"""
        mel_spec = self.mel_transform(audio)
        log_mel = torch.log(mel_spec + 1e-9)
        return log_mel

    def process_audio_file(self, file_path):
        """Process a single audio file to log-mel spectrogram"""
        audio = self.load_audio(file_path)
        mel_spec = self.compute_melspectrogram(audio)
        return mel_spec

    def normalize_spectrogram(self, spec):
        """Normalize spectrogram to zero mean and unit variance"""
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-9)

if __name__ == "__main__":
    file_path = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4/lipread_files/aufgaben/train/aufgaben_0282-0283.m4a"

    audio_processor = AudioProcessor()
    mel_spec = audio_processor.process_audio_file(file_path)
    normalized_spec = audio_processor.normalize_spectrogram(mel_spec)
    print(normalized_spec)
    print(normalized_spec.shape)