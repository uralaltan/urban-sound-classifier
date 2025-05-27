import argparse
import yaml
import torch
import csv
import logging
import time
from pathlib import Path
from datetime import datetime
import sys

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
from augment import spec_augment
from utils import seed_all, pad_collate
from models.baseline_cnn import BaselineCNN
from models.improved_acdnet import ImprovedACDNet
from models.ast_model import create_ast_model


def get_model(model_type, n_classes=10, **kwargs):
    if model_type == 'baseline':
        return BaselineCNN(n_classes)
    elif model_type == 'improved':
        return ImprovedACDNet(n_classes)
    elif model_type == 'baseline_cnn':
        return BaselineCNN(n_classes)
    elif model_type == 'se_cnn':
        return ImprovedACDNet(n_classes)
    elif model_type == 'ast_transformer':
        input_dims = kwargs.get('input_dims', (128, 173))
        patch_size = kwargs.get('patch_size', (16, 16))
        model_size = kwargs.get('model_size', 'base')
        dropout = kwargs.get('dropout', 0.1)
        pretrained = kwargs.get('pretrained', None)

        return create_ast_model(
            n_classes=n_classes,
            input_dims=input_dims,
            patch_size=patch_size,
            model_size=model_size,
            dropout=dropout,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, log_interval=50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
            elapsed = time.time() - start_time
            acc = 100. * correct / total
            avg_loss = total_loss / batch_idx
            logging.info(f'Epoch {epoch}, Step {batch_idx}/{len(train_loader)}, '
                         f'Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Time: {elapsed:.1f}s')

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def main(cfg_path, pretrained_path=None):
    cfg_file = (ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(open(cfg_file))

    seed_all(cfg['seed'])

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logging.info("Using CPU")

    csv_path = str((ROOT / cfg['csv']).resolve())
    audio_root = str((ROOT / cfg['audio_root']).resolve())

    train_ds = UrbanSoundDS(csv_path, audio_root,
                            folds=cfg['train_folds'],
                            transform=spec_augment)
    val_ds = UrbanSoundDS(csv_path, audio_root,
                          folds=cfg['val_folds'])

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
        batch_size=cfg['batch_size'] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=pad_collate
    )

    model_type = cfg.get('model_type', 'baseline')

    model_kwargs = {
        'input_dims': cfg.get('input_dims', (128, 173)),
        'patch_size': cfg.get('patch_size', (16, 16)),
        'model_size': cfg.get('model_size', 'base'),
        'dropout': cfg.get('dropout', 0.1),
        'pretrained': cfg.get('pretrained', None)
    }

    model = get_model(model_type, **model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {model_type}, Total params: {total_params:,}, Trainable: {trainable_params:,}")

    if pretrained_path:
        ckpt = (ROOT / pretrained_path).resolve()
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logging.info(f"Loaded pretrained weights from {ckpt.name}")

    weight_decay = cfg.get('weight_decay', 0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )

    label_smoothing = cfg.get('label_smoothing', 0.1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    results_csv = ROOT / "reports" / "figures" / f"results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"])

    best_acc = 0.0
    total_start_time = time.time()

    logging.info(f"Starting training for {cfg['epochs']} epochs...")

    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        logging.info(f'Epoch {epoch}/{cfg["epochs"]} - '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                     f'LR: {current_lr:.6f}, Time: {epoch_time:.1f}s')

        with open(results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_acc:.2f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"])

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = ROOT / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            fname = ckpt_dir / f"best_{model_type}_{val_acc:.2f}.pt"
            torch.save(model.state_dict(), fname)
            logging.info(f"New best model saved: {fname.name}")

    total_time = time.time() - total_start_time
    logging.info(f"Training complete! Best validation accuracy: {best_acc:.2f}%")
    logging.info(f"Total training time: {total_time / 60:.1f} minutes")
    logging.info(f"Results saved to: {results_csv.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast UrbanSound8K Training')
    parser.add_argument('--cfg', default='experiments/cfg_baseline.yaml',
                        help='Path to configuration file')
    parser.add_argument('--pretrained', default=None,
                        help='Path to pretrained checkpoint')

    args = parser.parse_args()
    main(args.cfg, args.pretrained)
