import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_logs(model_name):
    # =============================
    # UPDATED PATHS (Triple Modal)
    # =============================
    BASE_DIR = "/home/aswath/Projects/capstone/multimodel_lipread/audio_cues_video"
    csv_path = f"{BASE_DIR}/metrics/{model_name}_training_log.csv"
    plot_dir = f"{BASE_DIR}/plots"

    os.makedirs(plot_dir, exist_ok=True)

    # -----------------------------
    # Safety Check
    # -----------------------------
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    print(f"✅ Loading log file: {csv_path}")

    # -----------------------------
    # Load log file
    # -----------------------------
    df = pd.read_csv(csv_path)

    # Ensure epochs are valid
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
    df["epoch"] = df["epoch"].astype(int)

    epochs = df["epoch"]

    # -----------------------------
    # Plot Loss
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, df["val_loss"], label="Validation Loss", marker="s")
    plt.plot(epochs, df["test_loss"], label="Test Loss", marker="^")

    plt.title(f"{model_name} — Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    loss_path = os.path.join(plot_dir, f"{model_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(loss_path)
    print(f"✅ Saved loss plot → {loss_path}")
    plt.show()

    # -----------------------------
    # Plot Accuracy
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, df["val_acc"], label="Validation Accuracy", marker="s")
    plt.plot(epochs, df["test_acc"], label="Test Accuracy", marker="^")

    plt.title(f"{model_name} — Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    acc_path = os.path.join(plot_dir, f"{model_name}_accuracy_plot.png")
    plt.tight_layout()
    plt.savefig(acc_path)
    print(f"✅ Saved accuracy plot → {acc_path}")
    plt.show()


# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    model_name = "late_fusion_mobile"   # 🔁 CHANGE to your actual model name
    plot_logs(model_name)
