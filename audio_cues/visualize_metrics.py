import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(model_name):
    csv_path = f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics/{model_name}_training_log.csv"
    plot_dir = f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/plots"
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load CSV log into DataFrame
    df = pd.read_csv(csv_path)

    # Filter numeric epochs
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
    df["epoch"] = df["epoch"].astype(int)

    epochs = df["epoch"]

    # -----------------------------
    # Plot Loss
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df["train_loss"], label="Train Loss", marker="o", linestyle='-')
    plt.plot(epochs, df["val_loss"], label="Validation Loss", marker="s", linestyle='--')
    plt.plot(epochs, df["test_loss"], label="Test Loss", marker="^", linestyle=':')
    plt.title(f"{model_name} - Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(plot_dir, f"{model_name}_loss_plot.png")
    plt.tight_layout()
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    plt.show()

    # -----------------------------
    # Plot Accuracy
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df["train_acc"], label="Train Accuracy", marker="o", linestyle='-')
    plt.plot(epochs, df["val_acc"], label="Validation Accuracy", marker="s", linestyle='--')
    plt.plot(epochs, df["test_acc"], label="Test Accuracy", marker="^", linestyle=':')
    plt.title(f"{model_name} - Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(plot_dir, f"{model_name}_accuracy_plot.png")
    plt.tight_layout()
    plt.savefig(acc_path)
    print(f"Saved accuracy plot to {acc_path}")
    plt.show()


if __name__ == "__main__":
    model_name = "middle_fusion_resnet"
    plot_logs(model_name)
