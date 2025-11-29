# multimodal/dataset_triple.py
import os
import re
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer

from audio_data import GLipsDataset

SID_REGEX = re.compile(r"\d{4}-\d{4}")


class MultimodalTripleDataset(Dataset):
    """
    Strictly aligned dataset across (audio, cue, video/lip regions) by:
      (word, sequence_id, split)

    Guarantees:
      - Same spoken word
      - Same sequence id
      - Same split
      - Same class index (comes from audio dataset only)
    """

    def __init__(self,
                 root_dir,
                 cue_root,
                 lip_regions_root,
                 input_size=117,
                 split="train",
                 cue_mode="emotion",
                 embed_model="sentence-transformers/all-mpnet-base-v2",
                 cache_dir=".cache_cues"):

        self.split = split.lower()
        self.cue_root = cue_root
        self.lip_root = Path(lip_regions_root)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Base dataset for labels & class list (single source of truth)
        self.audio_ds = GLipsDataset(root_dir, input_size, split=self.split)
        self.classes = self.audio_ds.classes
        self.class_to_idx = self.audio_ds.class_to_idx
        self.cue_mode = cue_mode

        print("✅ Audio dataset loaded (source of class labels)")

        print("📖 Loading cue files...")
        self.cues = self._load_cues_by_split()

        print("📂 Indexing lip-region .npy files...")
        self.video_index = self._index_lip_regions()

        print("🔗 Strictly aligning modalities...")
        self.samples = self._align_modalities_strict()

        print("🧠 Loading SentenceTransformer...")
        self.embedder = SentenceTransformer(embed_model)

        print("💾 Caching embeddings...")
        self.desc2vec = self._cache_embeddings(embed_model)

    # =====================================================
    # CUE LOADING (matches your audio pipeline exactly)
    # =====================================================
    def _load_cues_by_split(self):
        folder = os.path.join(
            self.cue_root,
            f"Descriptions_{self.cue_mode.capitalize()}"
        )

        cues = {}
        for file in os.listdir(folder):
            fn = file.lower()

            if self.split not in fn:
                continue

            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)

            for entry in data:
                key = (
                    entry["word"],
                    entry["sequence_id"],
                    self.split
                )
                cues[key] = entry["description"]

        print(f"✅ Loaded {len(cues)} cue entries [{self.split}]")
        return cues

    # =====================================================
    # VIDEO INDEXER
    # =====================================================
    def _index_lip_regions(self):
        """
        Returns:
            dict[(word, sid, split)] = path-to-npy
        """
        index = {}

        for npy_file in self.lip_root.rglob("*.npy"):
            name = npy_file.name
            sid_match = SID_REGEX.search(name)
            if not sid_match:
                continue

            sid = sid_match.group()

            # infer split from full path
            parts = [p.lower() for p in npy_file.parts]
            if self.split not in parts:
                continue

            # infer word from folder
            word = None
            for cls in self.classes:
                if cls.lower() in parts:
                    word = cls
                    break

            if word is None:
                continue

            key = (word, sid, self.split)

            # enforce one-to-one mapping
            if key in index:
                raise RuntimeError(
                    f"Duplicate video entries for {key}:\n"
                    f"  Existing: {index[key]}\n"
                    f"  New:      {npy_file}"
                )

            index[key] = str(npy_file)

        print(f"✅ Indexed {len(index)} video samples")
        return index

    # =====================================================
    # STRICT ALIGNMENT
    # =====================================================
    def _align_modalities_strict(self):
        aligned = []

        skipped_video = 0
        skipped_cue = 0
        skipped_sid = 0

        for s in self.audio_ds.samples:

            audio_path = s["audio_path"]
            label = s["label"]
            word = self.audio_ds.classes[label]

            sid_match = SID_REGEX.search(audio_path)
            if not sid_match:
                skipped_sid += 1
                continue

            sid = sid_match.group()
            key = (word, sid, self.split)

            # cue must exist
            if key not in self.cues:
                skipped_cue += 1
                continue

            # video must exist
            if key not in self.video_index:
                skipped_video += 1
                continue

            aligned.append({
                "audio_path": audio_path,
                "label": label,
                "word": word,
                "sid": sid,
                "desc": self.cues[key],
                "lip_path": self.video_index[key],
            })

        print(
            f"\nAlignment summary [{self.split}]\n"
            f"  ✅ Aligned samples: {len(aligned)}\n"
            f"  ❌ Missing cues:    {skipped_cue}\n"
            f"  ❌ Missing video:   {skipped_video}\n"
            f"  ❌ Missing SID:     {skipped_sid}\n"
        )

        if len(aligned) == 0:
            raise RuntimeError("No aligned samples were built. Check folder structure and naming!")

        return aligned

    # =====================================================
    # EMBEDDING CACHE
    # =====================================================
    def _cache_embeddings(self, model_name):
        descs = sorted(set(s["desc"] for s in self.samples))
        sig = hashlib.md5("".join(descs).encode()).hexdigest()
        file = os.path.join(self.cache_dir, f"{self.cue_mode}_{sig}.npz")

        if os.path.exists(file):
            d = np.load(file, allow_pickle=True)
            print("✅ Loaded cached cue embeddings")
            return dict(zip(d["desc"], d["emb"]))

        print("⏳ Computing embeddings...")
        emb = self.embedder.encode(descs, show_progress_bar=True)
        np.savez(file, desc=descs, emb=emb)
        return dict(zip(descs, emb))

    # =====================================================
    # TORCH API
    # =====================================================
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # -------------------
        # AUDIO
        # -------------------
        mel = self.audio_ds.audio_processor.process_audio_file(s["audio_path"])
        mel = self.audio_ds.audio_processor.normalize_spectrogram(mel)
        mel = mel[:80, :self.audio_ds.input_size]
        mel = torch.tensor(mel, dtype=torch.float32)

        # -------------------
        # CUE
        # -------------------
        cue_vec = torch.tensor(self.desc2vec[s["desc"]], dtype=torch.float32)

        # -------------------
        # VIDEO (lip regions)
        # -------------------
        arr = np.load(s["lip_path"]).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # (T,H,W,C) -> (C,T,H,W)
        lip = torch.tensor(arr).permute(3, 0, 1, 2).float()

        # -------------------
        # LABEL
        # -------------------
        label = torch.tensor(s["label"], dtype=torch.long)

        return mel, cue_vec, lip, label


# =====================================================
# BATCH COLLATE
# =====================================================
def collate_fn_triple(batch):
    mels = torch.stack([b[0] for b in batch])
    cues = torch.stack([b[1] for b in batch])
    lips = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return mels, cues, lips, labels
