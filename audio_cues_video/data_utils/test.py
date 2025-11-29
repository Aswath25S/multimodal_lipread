# test_dataset_triple.py
import torch
import re
from dataset import MultimodalTripleDataset, collate_fn_triple

# ----------------------------
# CONFIGURE THESE PATHS
# ----------------------------
ROOT_DIR = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4"
CUE_ROOT = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4"
LIP_ROOT = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4_lip_regions"

INPUT_SIZE = 117
SPLIT = "train"
CUE_MODE = "emotion"   # or "environment"


def main():
    print("🔎 Initializing triple dataset...\n")

    ds = MultimodalTripleDataset(
        root_dir=ROOT_DIR,
        cue_root=CUE_ROOT,
        lip_regions_root=LIP_ROOT,
        input_size=INPUT_SIZE,
        split=SPLIT,
        cue_mode=CUE_MODE
    )

    print(f"\n✅ Dataset built successfully")
    print(f"Total aligned samples: {len(ds)}")
    print(f"Number of classes: {len(ds.classes)}")

    # ----------------------------
    # SAMPLING RESULTS
    # ----------------------------
    print("\n📌 Printing first 5 samples:")
    for i in range(min(5, len(ds))):
        s = ds.samples[i]
        print(f"\nSample {i}:")
        print("  Word :", s["word"])
        print("  SID  :", s["sid"])
        print("  Audio:", s["audio_path"])
        print("  Video:", s["lip_path"])
        print("  Cue  :", s["desc"][:80] + "...")

    # ----------------------------
    # SHAPE CHECK
    # ----------------------------
    print("\n📐 Checking tensor shapes:")
    mel, cue, lip, label = ds[0]

    print("Audio (mel):", mel.shape)       # (80, T)
    print("Cue vector:", cue.shape)        # (768,) or similar
    print("Video:", lip.shape)             # (C, T, H, W)
    print("Label:", label.item())

    assert mel.shape[0] == 80, "Mel dimension incorrect!"
    assert lip.dim() == 4, "Video tensor must be (C,T,H,W)"
    assert isinstance(label.item(), int), "Label must be integer"

    # ----------------------------
    # BATCH TEST
    # ----------------------------
    print("\n📦 Checking DataLoader batching:")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn_triple, shuffle=True)

    mel_b, cue_b, lip_b, lab_b = next(iter(loader))
    print("Batch audio:", mel_b.shape)
    print("Batch cue:", cue_b.shape)
    print("Batch video:", lip_b.shape)
    print("Batch labels:", lab_b.shape)

    # ----------------------------
    # CLASS CONSISTENCY
    # ----------------------------
    print("\n📛 Checking label-word consistency:")
    for i in range(10):
        s = ds.samples[i]
        label = s["label"]
        word = s["word"]
        assert ds.classes[label] == word, f"Mismatch at index {i}: label != word"

    print("✅ Class integrity verified")

    # ----------------------------
    # SID CONSISTENCY CHECK
    # ----------------------------
    print("\n🔗 Checking SID match in filenames:")
    sid_re = re.compile(r"\d{4}-\d{4}")

    for i in range(10):
        s = ds.samples[i]
        sid = s["sid"]
        assert sid in s["audio_path"], "SID mismatch in audio filename"
        assert sid in s["lip_path"], "SID mismatch in lip filename"

    print("✅ Sequence ID integrity verified")

    print("\n🎉 All checks PASSED!")


if __name__ == "__main__":
    main()
