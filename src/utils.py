import random, numpy as np, torch, os
import torch.nn.functional as F


def seed_all(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_collate(batch, max_frames=173):
    specs, labels = zip(*batch)
    # pad / truncate like above
    specs2 = []
    for mel in specs:
        T = mel.size(2)
        if T < max_frames:
            mel = F.pad(mel, (0, max_frames - T))
        else:
            mel = mel[:, :, :max_frames]
        specs2.append(mel)
    return torch.stack(specs2), torch.tensor(labels)
