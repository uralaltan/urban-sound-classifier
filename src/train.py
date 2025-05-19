import argparse, yaml, torch, os, wandb
from datasets import UrbanSoundDS
from augment import spec_augment
from utils import seed_all
from models.improved_acdnet import ImprovedACDNet


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])

    train_ds = UrbanSoundDS(cfg['csv'], cfg['audio_root'],
                            folds=cfg['train_folds'],
                            transform=spec_augment)
    val_ds = UrbanSoundDS(cfg['csv'], cfg['audio_root'],
                          folds=cfg['val_folds'])
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    model = ImprovedACDNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=cfg['lr'],
        steps_per_epoch=len(train_loader), epochs=cfg['epochs'])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    wandb.init(project='urbansound8k', config=cfg)
    best_acc = 0
    for epoch in range(cfg['epochs']):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            sched.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        wandb.log({'val_acc': acc, 'epoch': epoch})
        if acc > best_acc:
            best_acc = acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/best_{acc:.4f}.pt')
        print(f'Epoch {epoch + 1}/{cfg["epochs"]} - val_acc: {acc:.4f}')

    print(f'Best validation accuracy: {best_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='../experiments/cfg_improved.yaml')
    args = parser.parse_args()
    main(args.cfg)
