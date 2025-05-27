# Kentsel Ses Sınıflandırıcı

UrbanSound8K veri seti kullanarak kentsel çevresel ses sınıflandırması için esnek bir derin öğrenme çerçevesi. Basit konfigürasyon değişiklikleri ile üç farklı sinir ağı mimarisi arasında kolayca geçiş yapın.

## Hızlı Başlangıç

### Ön Gereksinimler

- Python 3.8+
- PyTorch 1.9+
- UrbanSound8K veri seti

### Kurulum

```bash
git clone https://github.com/uralaltan/urban-sound-classifier.git
cd urban-sound-classifier
pip install -r requirements.txt
```

### Model Eğitimi

Konfigürasyon dosyasını değiştirerek modeller arasında geçiş yapın:

```bash
# Temel CNN (hızlı, hafif)
python src/train.py --cfg experiments/cfg_baseline.yaml

# SE-CNN (dengeli performans)
python src/train.py --cfg experiments/cfg_improved.yaml

# AST Transformer (en yüksek doğruluk)
python src/train.py --cfg experiments/cfg_ast.yaml
```

## Model Karşılaştırması

| Model           | Doğruluk | Hız   | Kullanım Alanı |
| --------------- | -------- | ----- | -------------- |
| Temel CNN       | 79.4%    | Hızlı | Edge dağıtım   |
| SE-CNN          | 85.9%    | Orta  | Üretim         |
| AST Transformer | 90.5%    | Yavaş | Araştırma      |

## Proje Yapısı

```
src/
├── models/           # Model implementasyonları
├── train.py         # Ana eğitim scripti
├── datasets.py      # Veri yükleme
└── utils.py         # Yardımcı fonksiyonlar

experiments/         # Konfigürasyon dosyaları
├── cfg_baseline.yaml
├── cfg_improved.yaml
└── cfg_ast.yaml

data/
└── UrbanSound8K/   # Veri seti konumu
```

## Konfigürasyon

Her modelin `experiments/` klasöründe kendi YAML konfigürasyon dosyası vardır. Mimariler arasında geçiş yapmak için sadece `model_type` parametresini değiştirin:

- `baseline_cnn`: Hızlı çıkarım için hafif CNN
- `se_cnn`: Dikkat mekanizmalı Squeeze-and-Excitation CNN
- `ast_transformer`: Audio Spectrogram Transformer

## Veri Seti

UrbanSound8K veri setini [resmi siteden](https://urbansounddataset.weill.cornell.edu/urbansound8k.html) indirin ve `data/UrbanSound8K/` klasörüne çıkarın.

Veri seti 10 kentsel ses sınıfı içerir:

- Klima, Araba Kornası, Oynayan Çocuklar, Köpek Havlaması, Delme
- Motor Rölanti, Silah Sesi, Kırıcı, Siren, Sokak Müziği

## Araştırma Makalesi

Detaylı metodoloji, sonuçlar ve analiz için `reports/paper/conference_paper.md` dosyasındaki konferans bildirisine bakın.

## Gereksinimler

```
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
pyyaml>=5.4.0
librosa>=0.8.0
matplotlib>=3.4.0
```

## İletişim

**Ural Altan Bozkurt**  
Ankara Üniversitesi  
E-posta: 22290330@ogrenci.ankara.edu.tr
