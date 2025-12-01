import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(model_name):
    csv_path = f"./metrics/{model_name}_training_log.csv"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load CSV log into DataFrame
    df = pd.read_csv(csv_path)

    # Remove "FINAL" row (contains string "FINAL" instead of epoch)
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
    df["epoch"] = df["epoch"].astype(int)

    # -----------------------------
    # Plot Loss
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="o")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss", marker="o")

    plt.title("Training, Validation, and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/{model_name}_loss_plot.png")
    print("Saved loss plot to loss_plot.png")
    plt.show()

    # -----------------------------
    # Plot Accuracy
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker="o")
    plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy", marker="o")

    plt.title("Training, Validation, and Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/{model_name}_accuracy_plot.png")
    print("Saved accuracy plot to accuracy_plot.png")
    plt.show()


if __name__ == "__main__":
    models = ["resnet", "lstm_reset_attn", "lstm_resnet_trans", "resnet_lstm", "vgg_lstm", "vgg"]
    for model_name in models:
        plot_logs(model_name)
