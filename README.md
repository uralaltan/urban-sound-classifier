# Urban Sound Classifier

An end-to-end, reproducible pipeline for environmental sound classification on the UrbanSound8K dataset.

This project features:
- Transfer learning with PANNs CNN14
- Rich data augmentation (mixup, SpecAugment, noise injection)
- Advanced scheduler & optimizer techniques
- 10-fold cross-validation
- Model checkpointing & ensembling
- Clean, modular codebase

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/uralaltan/urban-sound-classifier.git
cd urban-sound-classifier
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

### 2. Download & Prepare UrbanSound8K

**macOS (with Homebrew):**
```bash
brew install wget
wget "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1" \
     -O UrbanSound8K.tar.gz
tar -xzf UrbanSound8K.tar.gz -C data/raw
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1" `
                  -OutFile "UrbanSound8K.tar.gz"
tar -xzf UrbanSound8K.tar.gz -C data\raw
```

Your directory should now contain `data/raw/UrbanSound8K/audio/...` and `data/raw/UrbanSound8K/metadata/UrbanSound8K.csv`.

### 3. Configure & Run Training

```bash
# Baseline experiment
python src/train.py --cfg experiments/cfg_baseline.yaml

# Improved experiment (transfer learning, stronger augmentations, etc.)
python src/train.py --cfg experiments/cfg_improved.yaml \
                    --noise_dir data/raw/noise \
                    --pretrained checkpoints/my_pretrained.pt
```

## 📂 Project Structure

```
urban-sound-classifier/
├── data/
│   ├── raw/                   # Original UrbanSound8K archive & optional noise files
│   └── processed/             # (Reserved) processed features, npy, etc.
├── experiments/
│   ├── cfg_baseline.yaml      # Simple CNN baseline settings
│   └── cfg_improved.yaml      # Transfer learning + rich augmentations
├── checkpoints/               # Saved model weights (best_{acc}.pt, foldX_epY_acc.pt)
├── reports/
│   ├── figures/               # val_results.csv, loss/acc plots
│   └── paper/                 # Your conference paper draft (LaTeX/docx)
├── src/
│   ├── augment.py             # mixup, SpecAugment, noise injection
│   ├── datasets.py            # PyTorch Dataset for UrbanSound8K
│   ├── train.py               # Main training & 10-fold CV + ensembling
│   ├── utils.py               # seed fixing, collate helpers
│   └── models/
│       ├── baseline_cnn.py    # 4-layer CNN baseline
│       ├── improved_acdnet.py # ACDNet + SE + multi-scale conv
│       └── transfer_acdnet.py # PANNs CNN14 backbone + custom head
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🔧 Configuration (YAML)

All hyperparameters live under `experiments/*.yaml`:

**cfg_baseline.yaml**
- Simple 3-block CNN
- 128 mel bands
- 30 epochs
- lr=1e-3
- label_smoothing=0.1

**cfg_improved.yaml**
- 256 mel bands
- Transfer learning
- mixup, strong SpecAugment, noise injection
- CosineAnnealingWarmRestarts (T₀=5)
- lr=5e-4
- weight_decay=1e-2
- label_smoothing=0.2

Edit any field (e.g. batch_size, epochs, device) to suit your machine.

## 🏃‍♂️ Running & Monitoring

Start training:
```bash
python src/train.py --cfg experiments/cfg_improved.yaml
```

**Optional flags:**
- `--noise_dir <path>`: directory of .wav noise files to inject
- `--pretrained <path>`: a saved checkpoint to fine-tune from

**Output:**
- Console logs: epoch, step, loss, validation accuracy, best checkpoints
- `reports/figures/val_results.csv`: fold, epoch, val_acc for all runs
- `checkpoints/`: saved .pt files named by fold/epoch/accuracy
- Final ensemble accuracy printed at the end of 10-fold CV

## 🎵 Dataset: UrbanSound8K

8732 labeled audio clips (≤4 s) in 10 urban sound classes:
```
air_conditioner, car_horn, children_playing, dog_bark, drilling,
engine_idling, gun_shot, jackhammer, siren, street_music
```

**Metadata**: `UrbanSound8K.csv` includes clip name, fold, start/end times, classID & class label.

**Folder layout:**
```
data/raw/UrbanSound8K/
├── audio/fold1/...fold10/   # WAV clips
└── metadata/UrbanSound8K.csv
```

## 🔍 Checkpoints & Ensembling

- Per-fold best: `checkpoints/fold{X}_ep{Y}_{acc:.4f}.pt`
- Top-3 ensemble: final evaluation averages softmax outputs of the top 3 models.
- Backup: existing val_results.csv is auto-renamed with a timestamp before each run.

## ⚖️ License & Citation

This code is released under the Apache 2.0 License.

If you use UrbanSound8K, please cite:
> J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", ACM Multimedia 2014.