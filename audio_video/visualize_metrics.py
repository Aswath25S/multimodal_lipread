import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(model_name):
    csv_path = f"/home/aswath/Projects/capstone/multimodel_lipread/audio_video/metrics/{model_name}_training_log.csv"
    plots_dir = "/home/aswath/Projects/capstone/multimodel_lipread/audio_video/plots/"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Load CSV log into DataFrame
    df = pd.read_csv(csv_path)

    # Ensure "epoch" column is numeric
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
    df["epoch"] = df["epoch"].astype(int)

    # -----------------------------
    # Plot Loss
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="o")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss", marker="o")

    plt.title(f"{model_name} - Training, Validation, and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(plots_dir, f"{model_name}_loss_plot.png")
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    plt.show()

    # -----------------------------
    # Plot Accuracy
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker="o")
    plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy", marker="o")

    plt.title(f"{model_name} - Training, Validation, and Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(plots_dir, f"{model_name}_accuracy_plot.png")
    plt.savefig(acc_path)
    print(f"Saved accuracy plot to {acc_path}")
    plt.show()


if __name__ == "__main__":
    # Example: change to your trained multimodal model
    models = ["early_fusion_fast", "early_fusion_mobilenet", "late_fusion_fast", "middle_fusion_fast", "middle_fusion_mobilenet"]
    for model in models:
        plot_logs(model)
