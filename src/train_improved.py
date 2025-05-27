import argparse
import yaml
import torch
import torch.nn.functional as F
import csv
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import UrbanSoundDS
from utils import seed_all, pad_collate
from models.custom_urbansound_net import CustomUrbanSoundNet, TransformerUrbanSoundNet, MixupLoss, mixup_data


class AdvancedSpecAugment:

    def __init__(self, freq_mask_prob=0.6, time_mask_prob=0.6,
                 freq_mask_size=12, time_mask_size=20):
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.freq_mask_size = freq_mask_size
        self.time_mask_size = time_mask_size

    def __call__(self, mel_spec):
        mel = mel_spec.clone()
        _, n_mels, time_steps = mel.shape

        if torch.rand(1) < self.freq_mask_prob:
            for _ in range(torch.randint(1, 3, (1,)).item()):
                freq_mask_start = torch.randint(0, max(1, n_mels - self.freq_mask_size), (1,))
                freq_mask_end = min(freq_mask_start + self.freq_mask_size, n_mels)
                mel[:, freq_mask_start:freq_mask_end, :] = mel.mean()

        if torch.rand(1) < self.time_mask_prob:
            for _ in range(torch.randint(1, 3, (1,)).item()):
                time_mask_start = torch.randint(0, max(1, time_steps - self.time_mask_size), (1,))
                time_mask_end = min(time_mask_start + self.time_mask_size, time_steps)
                mel[:, :, time_mask_start:time_mask_end] = mel.mean()

        return mel


class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


class EarlyStopping:

    def __init__(self, patience=25, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
                (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_epoch_advanced(model, train_loader, optimizer, criterion, mixup_criterion,
                         cfg, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if cfg.get('use_mixup', False) and torch.rand(1) < 0.5:
            mixed_data, target_a, target_b, lam = mixup_data(
                data, target, cfg.get('mixup_alpha', 0.3), device
            )
            optimizer.zero_grad()
            output = model(mixed_data)
            loss = mixup_criterion(output, target_a, target_b, lam)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

        loss.backward()

        if cfg.get('gradient_clipping', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clipping'])

        optimizer.step()

        total_loss += loss.item()
        if not cfg.get('use_mixup', False) or torch.rand(1) >= 0.5:
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    train_acc = 100. * correct / total if total > 0 else 0
    return total_loss / len(train_loader), train_acc


def validate_with_tta(model, val_loader, criterion, device, cfg):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    tta_steps = cfg.get('tta_steps', 1) if cfg.get('use_tta', False) else 1

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            if tta_steps > 1:
                outputs = []
                for _ in range(tta_steps):
                    augmented_data = data + torch.randn_like(data) * 0.01
                    output = model(augmented_data)
                    outputs.append(output)

                output = torch.stack(outputs).mean(dim=0)
            else:
                output = model(data)

            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def save_model(model, optimizer, scheduler, epoch, acc, loss, cfg, save_path, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__ if hasattr(scheduler, '__dict__') else None,
        'accuracy': acc,
        'loss': loss,
        'config': cfg
    }

    torch.save(checkpoint, save_path)
    if is_best:
        best_path = save_path.parent / f"best_{save_path.name}"
        torch.save(checkpoint, best_path)
        logging.info(f"New best model saved with accuracy: {acc:.2f}%")


def plot_training_history(history, save_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    if 'lr' in history:
        ax3.plot(epochs, history['lr'], 'go-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')

    if 'epoch_time' in history:
        ax4.plot(epochs, history['epoch_time'], 'mo-', label='Time per Epoch')
        ax4.set_title('Training Time per Epoch')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(cfg_path, pretrained_path=None):
    cfg_file = (ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(open(cfg_file))

    seed_all(cfg['seed'])

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")

    train_transform = AdvancedSpecAugment(
        freq_mask_prob=cfg.get('freq_mask_prob', 0.6),
        time_mask_prob=cfg.get('time_mask_prob', 0.6),
        freq_mask_size=cfg.get('freq_mask_size', 12),
        time_mask_size=cfg.get('time_mask_size', 20)
    ) if cfg.get('augmentation_prob', 0.8) > 0 else None

    csv_path = str((ROOT / cfg['csv']).resolve())
    audio_root = str((ROOT / cfg['audio_root']).resolve())

    train_ds = UrbanSoundDS(
        csv_path, audio_root,
        folds=cfg['train_folds'],
        transform=train_transform
    )

    val_ds = UrbanSoundDS(
        csv_path, audio_root,
        folds=cfg['val_folds'],
        transform=None
    )

    logging.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    num_workers = min(8, torch.get_num_threads())

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=pad_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=pad_collate
    )

    model_type = cfg.get('model_type', 'custom_urbansound')
    if model_type == 'custom_urbansound':
        model = CustomUrbanSoundNet(
            n_classes=10,
            dropout=cfg.get('dropout', 0.15)
        ).to(device)
    elif model_type == 'transformer_urbansound':
        model = TransformerUrbanSoundNet(
            n_classes=10,
            dropout=cfg.get('dropout', 0.15)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {model_type}")
    logging.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    if pretrained_path:
        ckpt = (ROOT / pretrained_path).resolve()
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logging.info(f"Loaded pretrained weights from {ckpt.name}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.02),
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=int(cfg.get('warmup_epochs', 10)),
        total_epochs=int(cfg['epochs']),
        base_lr=float(cfg['lr']),
        min_lr=float(cfg.get('min_lr', 1e-6))
    )

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=cfg.get('label_smoothing', 0.15)
    )
    mixup_criterion = MixupLoss(
        alpha=cfg.get('mixup_alpha', 0.3),
        label_smoothing=cfg.get('label_smoothing', 0.15)
    )

    early_stopping = EarlyStopping(
        patience=cfg.get('early_stopping_patience', 25),
        min_delta=cfg.get('early_stopping_min_delta', 0.001),
        mode='max'
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = ROOT / "reports" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / f"advanced_results_{timestamp}.csv"

    checkpoints_dir = ROOT / "models" / "checkpoints" / f"run_{timestamp}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"])

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'lr': [], 'epoch_time': []
    }

    best_acc = 0.0
    best_models = []
    total_start_time = time.time()

    logging.info(f"Starting advanced training for {cfg['epochs']} epochs...")
    logging.info(f"Target accuracy: 90%+")

    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()

        current_lr = scheduler.step()

        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, optimizer, criterion, mixup_criterion, cfg, device, epoch
        )

        val_loss, val_acc = validate_with_tta(model, val_loader, criterion, device, cfg)

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        logging.info(f'Epoch {epoch}/{cfg["epochs"]} - '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                     f'LR: {current_lr:.2e} - Time: {epoch_time:.1f}s')

        with open(results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time])

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        if cfg.get('save_top_k', 3) > 0:
            checkpoint_path = checkpoints_dir / f"model_epoch_{epoch}_acc_{val_acc:.2f}.pth"
            save_model(model, optimizer, scheduler, epoch, val_acc, val_loss, cfg, checkpoint_path, is_best)

            best_models.append((val_acc, checkpoint_path))
            best_models.sort(key=lambda x: x[0], reverse=True)

            if len(best_models) > cfg.get('save_top_k', 3):
                _, old_path = best_models.pop()
                if old_path.exists():
                    old_path.unlink()

        if cfg.get('save_every_n_epochs', 0) > 0 and epoch % cfg['save_every_n_epochs'] == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch}.pth"
            save_model(model, optimizer, scheduler, epoch, val_acc, val_loss, cfg, checkpoint_path)

        early_stopping(val_acc)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

        if val_acc >= 85.0 and val_acc < 90.0:
            logging.info(f"🎯 Reached {val_acc:.2f}% - Getting close to 90% target!")
        elif val_acc >= 90.0:
            logging.info(f"🏆 BREAKTHROUGH! Achieved {val_acc:.2f}% - Target reached!")

    total_time = time.time() - total_start_time
    logging.info(f"Training completed in {total_time / 3600:.2f} hours")
    logging.info(f"Best validation accuracy: {best_acc:.2f}%")

    final_path = checkpoints_dir / "final_model.pth"
    save_model(model, optimizer, scheduler, epoch, val_acc, val_loss, cfg, final_path)

    plot_path = results_dir / f"training_history_{timestamp}.png"
    plot_training_history(history, plot_path)
    logging.info(f"Training plots saved to {plot_path}")

    logging.info("\n" + "=" * 50)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Model: {model_type}")
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Training Samples: {len(train_ds)}")
    logging.info(f"Validation Samples: {len(val_ds)}")
    logging.info(f"Best Validation Accuracy: {best_acc:.2f}%")
    logging.info(f"Final Learning Rate: {current_lr:.2e}")
    logging.info(f"Total Training Time: {total_time / 3600:.2f} hours")
    logging.info(f"Average Time per Epoch: {np.mean(history['epoch_time']):.1f} seconds")
    logging.info(f"Results saved to: {results_csv}")
    logging.info(f"Model checkpoints: {checkpoints_dir}")

    if best_acc >= 90.0:
        logging.info("🏆 SUCCESS: Target accuracy of 90%+ achieved!")
    else:
        logging.info(f"📈 Progress: Reached {best_acc:.2f}% (Target: 90%+)")

    return best_acc, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced UrbanSound8K Training')
    parser.add_argument('--config', '-c', default='experiments/cfg_baseline.yaml',
                        help='Path to configuration file')
    parser.add_argument('--pretrained', '-p', default=None,
                        help='Path to pretrained model weights')
    parser.add_argument('--resume', '-r', default=None,
                        help='Path to checkpoint to resume training')

    args = parser.parse_args()

    try:
        best_accuracy, training_history = main(args.config, args.pretrained)
        logging.info(f"Training completed successfully with best accuracy: {best_accuracy:.2f}%")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise
