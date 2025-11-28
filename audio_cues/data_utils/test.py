# multimodal/test_dataset_mm.py

import torch
from dataset import MultimodalDataset


def main():
    ROOT = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4"       # same as in multimodal.yaml
    CUE_ROOT = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4"
    INPUT_SIZE = 117
    SPLIT = "train"

    print("\n🔍 Loading dataset...\n")

    dataset = MultimodalDataset(
    root_dir=ROOT,
    cue_root=CUE_ROOT,
    input_size=INPUT_SIZE,
    split=SPLIT,
    cue_mode="emotion"
)


    print("\n✅ Dataset Loaded")
    print("Total aligned samples:", len(dataset))

    # --------------------------------------------------
    # Inspect first 5 samples
    # --------------------------------------------------
    print("\n🧪 INSPECTING FIRST 5 SAMPLES...\n")

    for i in range(5):
        mel, cue, label = dataset[i]

        print(f"SAMPLE {i}")
        print("  Mel shape:", mel.shape)           # expected (80,117)
        print("  Cue shape:", cue.shape)           # expected (768,)
        print("  Label:", label.item())
        print("")

    # --------------------------------------------------
    # Full scan test
    # --------------------------------------------------
    print("\n🔁 FULL DATASET INTEGRITY CHECK...\n")

    missing = 0
    bad_shape = 0

    for i in range(len(dataset)):
        mel, cue, label = dataset[i]

        if mel.shape != (80, INPUT_SIZE):
            print("❌ Bad mel shape at idx", i, mel.shape)
            bad_shape += 1

        if cue.ndim != 1:
            print("❌ Bad cue embedding shape at idx", i, cue.shape)
            bad_shape += 1

    print("\n✅ Integrity check complete")
    print("Invalid shapes:", bad_shape)

    # --------------------------------------------------
    # Torch DataLoader test
    # --------------------------------------------------
    print("\n🚀 Testing DataLoader batch loading...\n")

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    specs, cues, labels = next(iter(loader))

    print("Batch Mel Shape:", specs.shape)   # (B,80,117)
    print("Batch Cue Shape:", cues.shape)    # (B,768)
    print("Batch Labels:", labels)



if __name__ == "__main__":
    main()
