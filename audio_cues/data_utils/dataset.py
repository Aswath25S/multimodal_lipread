# multimodal/dataset_mm.py
import os, json, re, hashlib
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

from .audio_data import GLipsDataset


class MultimodalDataset(Dataset):
    def __init__(self, root_dir, cue_root, input_size, split="train",
                 cue_mode="emotion",
                 embed_model="sentence-transformers/all-mpnet-base-v2",
                 cache_dir=".cache_cues"):

        assert cue_mode in ["emotion", "environment"], "cue_mode must be 'emotion' or 'environment'"

        self.audio_ds = GLipsDataset(root_dir, input_size, split=split)
        self.split = split.lower()
        self.cue_root = cue_root
        self.cache_dir = cache_dir
        self.cue_mode = cue_mode
        os.makedirs(cache_dir, exist_ok=True)

        print(f"📂 Using cues from: Descriptions_{cue_mode.capitalize()}")

        # Load cues ONLY from selected mode
        print("🔍 Loading JSON cue data...")
        self.cues = self._load_jsons_mode()

        # Align audio with cue
        print("🔗 Aligning audio and cues...")
        self.samples = self._build_pairs()

        # Sentence Transformer model
        print("🧠 Loading sentence-transformer...")
        self.embedder = SentenceTransformer(embed_model)

        # Cache embeddings only for aligned samples
        print("💾 Preparing embedding cache...")
        self.desc2vec = self._cache_embeddings(embed_model)

    # ------------------------------------------------------
    def _load_jsons_mode(self):
        cue_index = {}

        folder = os.path.join(self.cue_root, f"Descriptions_{self.cue_mode.capitalize()}")

        for file in os.listdir(folder):
            fname = file.lower()

            # detect split
            if "train" in fname: split = "train"
            elif "val" in fname: split = "val"
            elif "test" in fname: split = "test"
            else: continue

            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)

            for entry in data:
                w   = entry["word"]
                sid = entry["sequence_id"]
                d   = entry["description"]
                cue_index[(w, sid, split)] = d

        print(f"✅ Loaded {len(cue_index)} {self.cue_mode} cues")
        return cue_index

    # ------------------------------------------------------
    def _build_pairs(self):
        aligned = []

        for s in self.audio_ds.samples:
            audio_path = s["audio_path"]
            label = s["label"]
            word = self.audio_ds.classes[label]

            # extract sequence id from filename
            match = re.search(r"\d{4}-\d{4}", audio_path)
            if not match:
                print("⚠️ No seq_id in filename:", audio_path)
                continue

            sid = match.group()
            key = (word, sid, self.split)

            if key not in self.cues:
                continue

            aligned.append({
                "audio_path": audio_path,
                "label": label,
                "desc": self.cues[key],
                "word": word,
                "sid": sid
            })

        print(f"✅ Final aligned samples [{self.split} | {self.cue_mode}]: {len(aligned)}")
        return aligned

    # ------------------------------------------------------
    def _cache_embeddings(self, model_name):
        descs = sorted(set(s["desc"] for s in self.samples))
        sig = hashlib.md5("".join(descs).encode()).hexdigest()

        cache_file = os.path.join(self.cache_dir, f"{self.cue_mode}_{sig}.npz")

        if os.path.exists(cache_file):
            print("✅ Loaded cached embeddings")
            data = np.load(cache_file, allow_pickle=True)
            return dict(zip(data["desc"], data["emb"]))

        print("⏳ Computing text embeddings...")
        emb = self.embedder.encode(descs, show_progress_bar=True)

        np.savez(cache_file, desc=descs, emb=emb)
        print("✅ Cached embeddings")

        return dict(zip(descs, emb))

    # ------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------
    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- AUDIO ----
        mel = self.audio_ds.audio_processor.process_audio_file(s["audio_path"])
        mel = self.audio_ds.audio_processor.normalize_spectrogram(mel)
        mel = mel[:80, :self.audio_ds.input_size]

        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel, dtype=torch.float32)
        else:
            mel = mel.clone().detach()

        # ---- CUE EMBEDDING ----
        vec = self.desc2vec[s["desc"]]
        if not isinstance(vec, torch.Tensor):
            cue_vec = torch.from_numpy(vec).float()
        else:
            cue_vec = vec.clone().detach()

        # ---- LABEL ----
        label = s["label"]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach()

        return mel, cue_vec, label


def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.tensor([b[2] for b in batch]))