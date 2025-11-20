import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(model_name):
    csv_path = f"./metrics/{model_name}_training_log.csv"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load CSV log
    df = pd.read_csv(csv_path)

    # Ensure "epoch" column is numeric
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
    df["epoch"] = df["epoch"].astype(int)

    # Create plots directory
    os.makedirs("./plots", exist_ok=True)

    # -----------------------------
    # Plot Loss (Train + Val)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="o")

    plt.title(f"{model_name} — Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = f"./plots/{model_name}_loss_plot.png"
    plt.savefig(out_path)
    print(f"Saved loss plot to {out_path}")
    plt.show()

    # -----------------------------
    # Plot Accuracy (Train + Val)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker="o")

    plt.title(f"{model_name} — Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = f"./plots/{model_name}_accuracy_plot.png"
    plt.savefig(out_path)
    print(f"Saved accuracy plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    model_name = "dense_nn"
    plot_logs(model_name)