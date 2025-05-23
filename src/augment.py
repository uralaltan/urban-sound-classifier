import random


def spec_augment(mel_spec,
                 time_mask_param: int = 20,
                 freq_mask_param: int = 10,
                 num_masks: int = 2):
    augmented = mel_spec.clone()
    n_mels, time_steps = augmented.shape[1], augmented.shape[2]

    for _ in range(num_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        augmented[:, f0:f0 + f, :] = 0

    for _ in range(num_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, time_steps - t))
        augmented[:, :, t0:t0 + t] = 0

    return augmented
