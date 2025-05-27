import torch
import torch.nn.functional as F


def spec_augment(mel_spec, freq_mask_prob=0.3, time_mask_prob=0.3,
                 freq_mask_size=8, time_mask_size=16):
    if torch.rand(1) > 0.5:
        return mel_spec

    mel = mel_spec.clone()
    _, n_mels, time_steps = mel.shape

    if torch.rand(1) < freq_mask_prob:
        freq_mask_start = torch.randint(0, max(1, n_mels - freq_mask_size), (1,))
        freq_mask_end = min(freq_mask_start + freq_mask_size, n_mels)
        mel[:, freq_mask_start:freq_mask_end, :] = mel.mean()

    if torch.rand(1) < time_mask_prob:
        time_mask_start = torch.randint(0, max(1, time_steps - time_mask_size), (1,))
        time_mask_end = min(time_mask_start + time_mask_size, time_steps)
        mel[:, :, time_mask_start:time_mask_end] = mel.mean()

    return mel


def minimal_augment(mel_spec):
    if torch.rand(1) > 0.3:
        return mel_spec

    noise = torch.randn_like(mel_spec) * 0.01
    return mel_spec + noise


def no_augment(mel_spec):
    return mel_spec
