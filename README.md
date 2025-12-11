# Multimodal Lip Reading

Multimodal lip reading on the GLips dataset using **audio**, **video**, **audio–video fusion**, and **textual cue–augmented** models.

The repository contains separate pipelines for:

- **Audio-only** speech classification from log-mel spectrograms.
- **Video-only** visual speech recognition from lip-region frame sequences.
- **Audio–video (AV) fusion** models with early/mid/late fusion variants.
- **Audio + textual cues**, **video + textual cues**, and **audio + video + textual cues** fusion models.

## Repository Structure

```text
multimodel_lipread/
├── audio/               # Audio-only models, datasets, configs, training
│   ├── configs/         # YAML configs for audio models
│   ├── data_utils/      # GLips audio dataset + loaders
│   ├── models/          # Audio CNN/LSTM/Transformer variants
│   ├── utils/           # Audio preprocessing (mel-spectrograms, normalization)
│   └── train.py         # Audio training script
├── video/               # Video-only (lip reading) models and pipeline
│   ├── config/          # YAML configs for visual models
│   ├── data_utils/      # Lip-region dataset + preprocessing
│   ├── models/          # ResNet/VGG/ShuffleNet/MobileNet variants
│   └── train.py         # Visual training script
├── audio_video/         # Multimodal audio–video fusion models
│   ├── config/          # YAML config for AV models
│   ├── data_utils/      # GLips multimodal dataset loader
│   ├── models/          # Early/mid/late fusion architectures
│   └── train.py         # AV training script
├── audio_cues/          # Audio + textual cue fusion (mel + descriptions)
│   ├── configs/         # YAML configs for audio+cue experiments
│   ├── data_utils/      # Dataset + loaders for mel + cue embeddings
│   ├── models/          # MobileNet / ResNet cue-fusion variants
│   └── train.py         # Audio + cue training script
├── cues_video/          # Video + textual cue fusion (lip regions + descriptions)
│   ├── configs/         # YAML configs for video+cue experiments
│   ├── data_utils/      # Dataset + loaders for lip regions + cue embeddings
│   ├── models/          # MobileNet / ResNet cue-fusion variants
│   └── train.py         # Video + cue training script
├── audio_cues_video/    # Joint audio + video + textual cue fusion
│   ├── configs/         # YAML configs for triple-modality experiments
│   ├── data_utils/      # Dataset utilities for (audio, video, cue)
│   ├── models/          # Early/mid/late triple-fusion architectures
│   └── train.py         # Audio + video + cue training script
├── cues/                # Utilities for generating / handling textual cues
├── data/                # Expected root for GLips dataset (not versioned)
├── data_clean.py        # Script for cleaning cue description JSON files
├── requirements.txt     # Python dependencies (audio, video, fusion, cues)
├── venv/                # Optional local virtual environment (not required)
└── README.md            # This file
```

Paths to datasets, cues, and hyperparameters are configured via the YAML files in
`audio/configs/`, `video/config/`, `audio_video/config/`, `audio_cues/configs/`,
`cues_video/configs/`, and `audio_cues_video/configs/`.

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

### 4. Audio + textual cue models

Audio + cue models use mel-spectrograms together with textual descriptions
(encoded as sentence embeddings) for each utterance.

1. Edit `audio_cues/configs/ac_config.yaml` to set:

   - `dataset.root_dir` – GLips root for audio files
   - `dataset.cue_root` – root directory containing JSON cue descriptions
   - `dataset.input_size` – mel feature dimension
   - `dataset.num_classes`
   - `train.*` (batch size, epochs, learning rate, etc.)

2. Run training:

   ```bash
   python audio_cues/train.py
   ```

   Metrics and checkpoints are written under
   `audio_cues/metrics/` and `audio_cues/models_trained/`.

### 5. Video + textual cue models

Video + cue models fuse lip-region sequences with textual descriptions.

1. Edit `cues_video/configs/cv_config.yaml` to set:

   - `dataset.cue_root` – cue JSON root
   - `dataset.lip_regions_root` – precomputed lip-region `.npy` files
   - `train.model_name` – fusion architecture (e.g., `early_fusion_mobile`)
   - `train.*` hyperparameters

2. Run training:

   ```bash
   python cues_video/train.py
   ```

   Logs are saved under `cues_video/metrics/` and models under
   `cues_video/models_trained/`.

### 6. Audio + video + textual cue models

Triple-modality models jointly use audio, lip regions, and textual cues.

1. Edit `audio_cues_video/configs/acv_config.yaml` to set:

   - `dataset.root_dir` – GLips root for audio/video
   - `dataset.cue_root` – cue JSON root
   - `dataset.lip_regions_root` – lip-region `.npy` directory
   - `dataset.input_size` – audio feature dimension
   - `dataset.num_classes`
   - `dataset.cue_mode`, `dataset.embed_model`, `dataset.cache_dir` (cue settings)
   - `train.*` hyperparameters (batch, lr, epochs, save/metrics dirs)

2. Run training:

   ```bash
   python audio_cues_video/train.py
   ```

   Metrics/checkpoints are written to the directories configured in
   `acv_config.yaml` (typically `./metrics` and `./models_trained`).

## Cue Data Cleaning

The `data_clean.py` helper script sanitizes cue description JSON files by
replacing explicit occurrences of the target word with a placeholder.
This ensures descriptions do not trivially leak the label.

Example usage (paths are hard-coded in the script and can be adjusted):

```bash
python data_clean.py
```

It reads a GLips description JSON, rewrites the descriptions with
`"target word"` placeholders, and writes a corrected JSON in a separate
directory.

## Notes

- Training is designed for GPU (CUDA) if available; the scripts
  automatically fall back to CPU otherwise.
- To change architectures, modify `model.name` or `train.model_name` in the
  appropriate YAML config.
- All training scripts log per-epoch train/val/test loss and accuracy and
  save best checkpoints for later evaluation.
