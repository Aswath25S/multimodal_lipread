import os
import re
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

SID_REGEX = re.compile(r"\d{4}-\d{4}")


class MultimodalCueVideoDataset(Dataset):
    """
    Strict cue + video dataset aligned by:
        (word, sequence_id, split)
    """

    def __init__(
        self,
        cue_root,
        lip_regions_root,
        split="train",
        cue_mode="emotion",
        embed_model="sentence-transformers/all-mpnet-base-v2",
        cache_dir=".cache_cues"
    ):
        self.split = split.lower()
        self.cue_root = cue_root
        self.lip_root = lip_regions_root
        self.cache_dir = cache_dir
        self.cue_mode = cue_mode

        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"\n🧪 Testing dataset split = {self.split}")
        print("📖 Loading cues...")
        self.cues = self._load_cues_by_split()

        print("📂 Indexing videos...")
        self.video_index = self._index_videos()

        print("🔗 Aligning modalities...")
        self.samples = self._align()

        print("🧠 Loading sentence model...")
        self.embedder = SentenceTransformer(embed_model, device="cpu")

        print("💾 Caching embeddings...")
        self.desc2vec = self._cache_embeddings(embed_model)

        print(f"✅ Dataset ready with {len(self.samples)} samples\n")

    # ------------------------------------
    # LOAD CUES
    # ------------------------------------
    def _load_cues_by_split(self):
        folder = os.path.join(self.cue_root, f"Descriptions_{self.cue_mode.capitalize()}")
        cues = {}

        for file in os.listdir(folder):
            if self.split not in file.lower():
                continue

            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)

            for entry in data:
                key = (entry["word"], entry["sequence_id"], self.split)
                cues[key] = entry["description"]

        print(f"✅ Loaded {len(cues)} cue entries [{self.split}]")
        return cues

    # ------------------------------------
    # INDEX VIDEOS
    # ------------------------------------
    def _index_videos(self):
        index = {}
        count = 0

        for word in os.listdir(self.lip_root):
            word_dir = os.path.join(self.lip_root, word)
            if not os.path.isdir(word_dir):
                continue

            split_dir = os.path.join(word_dir, self.split)
            if not os.path.isdir(split_dir):
                continue

            for fname in os.listdir(split_dir):
                if not fname.endswith(".npy"):
                    continue

                sid_match = SID_REGEX.search(fname)
                if not sid_match:
                    print(f"⚠️ Bad filename (SID not found): {fname}")
                    continue

                sid = sid_match.group()
                key = (word, sid, self.split)
                path = os.path.join(split_dir, fname)

                if key in index:
                    print("❌ DUPLICATE ENTRY FOUND")
                    print("Existing:", index[key])
                    print("New     :", path)
                    raise RuntimeError(f"Duplicate video for {key}")

                index[key] = path
                count += 1

        print(f"✅ Indexed {count} videos [{self.split}]")
        return index

    # ------------------------------------
    # ALIGN CUES + VIDEOS
    # ------------------------------------
    def _align(self):
        aligned = []
        miss_cue = 0
        miss_vid = 0

        for key in self.cues:
            if key not in self.video_index:
                miss_vid += 1
                continue

            aligned.append({
                "word": key[0],
                "sid": key[1],
                "split": key[2],
                "desc": self.cues[key],
                "lip_path": self.video_index[key]
            })

        print(
            f"\nAlignment summary [{self.split}]\n"
            f"  ✅ Aligned samples: {len(aligned)}\n"
            f"  ❌ Missing videos: {miss_vid}\n"
        )

        if len(aligned) == 0:
            raise RuntimeError("No valid samples after alignment!")

        return aligned

    # ------------------------------------
    # CACHE EMBEDDINGS
    # ------------------------------------
    def _cache_embeddings(self, model_name):
        descs = sorted(set(s["desc"] for s in self.samples))
        sig = hashlib.md5("".join(descs).encode()).hexdigest()
        path = os.path.join(self.cache_dir, f"{self.cue_mode}_{sig}.npz")

        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            print("✅ Loaded cached cue embeddings")
            return dict(zip(d["desc"], d["emb"]))

        print("⏳ Computing embeddings...")
        emb = self.embedder.encode(descs, show_progress_bar=True)
        np.savez(path, desc=descs, emb=emb)
        return dict(zip(descs, emb))

    # ------------------------------------
    # DATASET API
    # ------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # CUE
        cue = torch.tensor(self.desc2vec[s["desc"]], dtype=torch.float32)

        # VIDEO
        arr = np.load(s["lip_path"]).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # (T,H,W,C) → (C,T,H,W)
        video = torch.tensor(arr).permute(3, 0, 1, 2)

        label = s["word"]
        sid = s["sid"]

        return cue, video, label, sid
