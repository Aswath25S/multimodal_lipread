import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_logs(model_name):
    csv_path = f"/home/aswath/Projects/capstone/multimodel_lipread/cues_video/metrics/{model_name}_training_log.csv"

    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    # Load CSV
    df = pd.read_csv(csv_path)

    # Convert epoch to numeric
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    df["epoch"] = df["epoch"].astype(int)

    # Required columns
    required = [
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc"
    ]

    for col in required:
        if col not in df.columns:
            print(f"⚠️ Missing column: {col}")
            return

    # Create plots directory
    os.makedirs("./plots", exist_ok=True)

    # -----------------------------
    # LOSS CURVES
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss", marker="o")

    plt.title(f"{model_name} — Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    loss_path = f"./plots/{model_name}_loss.png"
    plt.savefig(loss_path, dpi=200)
    print(f"✅ Saved: {loss_path}")
    plt.close()

    # -----------------------------
    # ACCURACY CURVES
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc", marker="o")
    plt.plot(df["epoch"], df["test_acc"], label="Test Acc", marker="o")

    plt.title(f"{model_name} — Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    acc_path = f"/home/aswath/Projects/capstone/multimodel_lipread/cues_video/plots/{model_name}_accuracy.png"
    plt.savefig(acc_path, dpi=200)
    print(f"✅ Saved: {acc_path}")
    plt.close()

    # -----------------------------
    # OVERFITTING DIAGNOSTIC
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"] - df["val_acc"], marker="o")

    plt.title(f"{model_name} — Train vs Val Gap")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Gap (%)")
    plt.axhline(0, linestyle="--")
    plt.grid(True)
    plt.tight_layout()

    gap_path = f"/home/aswath/Projects/capstone/multimodel_lipread/cues_video/plots/{model_name}_overfit_gap.png"
    plt.savefig(gap_path, dpi=200)
    print(f"✅ Saved: {gap_path}")
    plt.close()

    print("\n✅ All plots generated successfully.")


if __name__ == "__main__":
    model_name = "cue_video_model"   # MUST MATCH TRAIN.PY
    plot_logs(model_name)
