# augment.py - Lightweight augmentation for faster training
import torch
import torch.nn.functional as F


def spec_augment(mel_spec, freq_mask_prob=0.3, time_mask_prob=0.3,
                 freq_mask_size=8, time_mask_size=16):
    """
    Lightweight SpecAugment implementation for faster training
    Reduces augmentation intensity for speed while maintaining effectiveness
    """
    if torch.rand(1) > 0.5:  # Apply augmentation 50% of the time
        return mel_spec

    mel = mel_spec.clone()
    _, n_mels, time_steps = mel.shape

    # Frequency masking (reduce probability and size for speed)
    if torch.rand(1) < freq_mask_prob:
        freq_mask_start = torch.randint(0, max(1, n_mels - freq_mask_size), (1,))
        freq_mask_end = min(freq_mask_start + freq_mask_size, n_mels)
        mel[:, freq_mask_start:freq_mask_end, :] = mel.mean()

    # Time masking (reduce probability and size for speed)
    if torch.rand(1) < time_mask_prob:
        time_mask_start = torch.randint(0, max(1, time_steps - time_mask_size), (1,))
        time_mask_end = min(time_mask_start + time_mask_size, time_steps)
        mel[:, :, time_mask_start:time_mask_end] = mel.mean()

    return mel


def minimal_augment(mel_spec):
    """
    Minimal augmentation for fastest training
    Only applies simple noise addition
    """
    if torch.rand(1) > 0.3:  # Apply only 30% of the time
        return mel_spec

    # Add small amount of noise
    noise = torch.randn_like(mel_spec) * 0.01
    return mel_spec + noise


def no_augment(mel_spec):
    """
    No augmentation - fastest option
    """
    return mel_spec
