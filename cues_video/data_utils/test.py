from dataset import MultimodalCueVideoDataset


def test_dataset(split="train"):
    ds = MultimodalCueVideoDataset(
        cue_root="/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4/",
        lip_regions_root="/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4_lip_regions/lipread_files/",
        split=split,
        cue_mode="emotion"
    )

    print(f"\n🔎 Inspecting first 5 samples [{split}]")

    for i in range(min(5, len(ds))):
        cue, vid, label, sid = ds[i]
        print(f"\nSample {i}")
        print("Word:", label)
        print("SID :", sid)
        print("Cue shape:", cue.shape)
        print("Video shape:", vid.shape)

    print("\n✅ PASSED DATASET TEST\n")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        test_dataset(split)
