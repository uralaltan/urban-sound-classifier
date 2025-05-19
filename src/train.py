import argparse
import yaml
import torch
import csv
import logging
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
from utils import seed_all
from models.improved_acdnet import ImprovedACDNet


def main(cfg_path, pretrained_path=None):
    cfg_file = (ROOT / cfg_path).resolve()
    cfg = yaml.safe_load(open(cfg_file))

    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])

    csv_path = str((ROOT / cfg['csv']).resolve())
    audio_root = str((ROOT / cfg['audio_root']).resolve())
    train_ds = UrbanSoundDS(csv_path, audio_root,
                            folds=cfg['train_folds'],
                            transform=spec_augment)
    val_ds = UrbanSoundDS(csv_path, audio_root,
                          folds=cfg['val_folds'])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=4
    )

    model = ImprovedACDNet().to(device)

    if pretrained_path:
        ckpt = (ROOT / pretrained_path).resolve()
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logging.info(f"Loaded pretrained weights from {ckpt.name}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=cfg['lr'],
        steps_per_epoch=len(train_loader),
        epochs=cfg['epochs'])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    results_csv = ROOT / "reports" / "figures" / "val_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    if results_csv.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = results_csv.parent / f"val_results_{ts}.csv"
        results_csv.rename(backup)
        logging.info(f"Existing results CSV renamed to {backup.name}")

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_accuracy"])

    best_acc = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        logging.info(f"Starting epoch {epoch}/{cfg['epochs']}")
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            sched.step()

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                logging.info(f"Epoch {epoch}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        logging.info(f"Epoch {epoch} validation accuracy: {val_acc:.4f}")

        with open(results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{val_acc:.4f}"])

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = ROOT / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            fname = ckpt_dir / f"best_{best_acc:.4f}.pt"
            torch.save(model.state_dict(), fname)
            logging.info(f"New best model saved: {fname.name}")

    logging.info(f"Training complete. Best val_acc: {best_acc:.4f}")
    logging.info(f"Results logged to: {results_csv.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='experiments/cfg_improved.yaml')
    p.add_argument('--pretrained', default=None,
                   help="Path under project root to a .pt checkpoint to load")
    args = p.parse_args()
    main(args.cfg, args.pretrained)
