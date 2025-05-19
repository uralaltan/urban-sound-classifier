import pandas as pd
import torchaudio
import torch
import torch.nn.functional as F
from pathlib import Path


class UrbanSoundDS(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_root, folds, transform=None,
                 sample_rate: int = 22050, n_meals: int = 128, max_frames=173):
        meta = pd.read_csv(csv_path)
        self.df = meta[meta['fold'].isin(folds)].reset_index(drop=True)
        self.audio_root = Path(audio_root)
        self.transform = transform
        self.target_sr = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024,
            hop_length=512, n_mels=n_meals,
            f_min=20.0, f_max=8000.0, power=2.0
        )
        self.max_frames = max_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = self.audio_root / f"fold{row.fold}" / row.slice_file_name
        waveform, sr = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        mel = self.mel(waveform).clamp(min=1e-9).log()
        T = mel.size(2)
        if T < self.max_frames:
            mel = F.pad(mel, (0, self.max_frames - T))
        else:
            mel = mel[:, :, :self.max_frames]
        if self.transform:
            mel = self.transform(mel)
        return mel, row.classID
