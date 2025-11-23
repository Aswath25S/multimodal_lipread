# Multimodal Lip Reading

Multimodal lip reading on the GLips dataset using **audio**, **video**, and **audio–video fusion** models.

The repository contains separate pipelines for:

- **Audio-only** speech classification from log-mel spectrograms.
- **Video-only** visual speech recognition from lip-region frame sequences.
- **Audio–video (AV) fusion** models with early/mid/late fusion variants.

## Repository Structure

```text
multimodel_lipread/
├── audio/              # Audio-only models, datasets, configs, training
│   ├── configs/        # YAML configs for audio models
│   ├── data_utils/     # GLips audio dataset + loaders
│   ├── models/         # Audio CNN/LSTM/Transformer variants
│   ├── utils/          # Audio preprocessing (mel-spectrograms, normalization)
│   └── train.py        # Audio training script
├── video/              # Video-only (lip reading) models and pipeline
│   ├── config/         # YAML configs for visual models
│   ├── data_utils/     # Lip-region dataset + preprocessing
│   ├── models/         # ResNet/VGG/ShuffleNet/MobileNet variants
│   └── train.py        # Visual training script
├── audio_video/        # Multimodal audio–video fusion models
│   ├── config/         # YAML config for AV models
│   ├── data_utils/     # GLips multimodal dataset loader
│   ├── models/         # Early/mid/late fusion architectures
│   └── train.py        # AV training script
├── cues/               # Additional utilities / cues used in experiments
├── data/               # Expected root for GLips dataset (not versioned)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

Paths to dataset and hyperparameters are configured via the YAML files in
`audio/configs/`, `video/config/`, and `audio_video/config/`.

## Setup

1. **Clone and enter the repo**

   ```bash
   git clone <repository-url>
   cd multimodel_lipread
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate         # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Data

The code assumes the **GLips** dataset is available under `data/`, e.g.

```text
multimodel_lipread/
└── data/
    └── GLips_40/           # or GLips_8, etc.
        └── lipread_files/
            ├── WORD_1/
            │   ├── train/*.m4a / *.mp4
            │   ├── val/*.m4a / *.mp4
            │   └── test/*.m4a / *.mp4
            └── WORD_2/
                └── ...
```

Adjust the `dataset.root_dir` entries in the YAML configs to point to your
GLips root directory.

## Visual Preprocessing (Lip Regions)

For video models, you must first extract lip-region sequences from raw videos.

1. Set `dataset.root_dir` and preprocessing options in:

   ```text
   video/config/visual_config.yaml
   ```

2. Run the preprocessing script:

   ```bash
   python -m video.data_utils.visual_preprocessing
   ```

   This will process all `.mp4` files under `dataset.root_dir`, extract lip
   sequences, and save them as `.npy` files. The visual `DataLoader` expects
   these preprocessed lip-region files to exist before training.

## Training

### 1. Audio-only models

1. Edit `audio/configs/audio_config.yaml` to set:

   - `dataset.root_dir` (GLips root)
   - `dataset.num_classes`
   - `dataset.input_size`
   - `model.name`, `model.version`
   - `training.*` (batch size, learning rate, epochs, etc.)

2. Run audio training:

   ```bash
   python audio/train.py
   ```

   Metrics are logged under `./metrics/` and trained weights under
   `./models_trained/`.

### 2. Video-only models

1. Edit `video/config/visual_config.yaml` to set dataset path, model name,
   and training hyperparameters.

2. Ensure visual preprocessing has been run (see section above).

3. Run visual training:

   ```bash
   python video/train.py
   ```

   Logs are written to `./metrics/` and checkpoints to the directory specified
   by `training.save_dir` in the visual config.

### 3. Audio–video fusion models

1. Edit `audio_video/config/av_config.yaml` to set:

   - `dataset.root_dir`
   - `dataset.audio_input_size`
   - `dataset.num_classes`
   - `model.name` (e.g., `early_fusion_mobilenet`, `middle_fusion_fast`, etc.)
   - `training.*` hyperparameters

2. Run AV training:

   ```bash
   python audio_video/train.py
   ```

   Best-performing models are saved under `./models_trained/` and metrics under
   `./metrics/`.

## Notes

- Training is designed for GPU (CUDA) if available; the scripts
  automatically fall back to CPU otherwise.
- To change architectures, modify `model.name` in the appropriate YAML config.
- All training scripts log per-epoch train/val/test loss and accuracy and
  save best checkpoints for later evaluation.

