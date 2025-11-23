import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(model_name):
    csv_path = f"/home/aswath/Projects/capstone/multimodel_lipread/video/metrics/{model_name}_training_log.csv"

    if not os.path.exists(csv_path):
        print(f"[ERROR] Log file not found: {csv_path}")
        return

    os.makedirs("/home/aswath/Projects/capstone/multimodel_lipread/video/plots", exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Remove any non-epoch rows
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
    df["epoch"] = df["epoch"].astype(int)

    if df.empty:
        print("[ERROR] No valid epoch rows found in CSV.")
        return

    # ===========================
    # LOSS PLOT
    # ===========================
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss", marker="o")

    plt.title(f"{model_name} - Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    loss_path = f"/home/aswath/Projects/capstone/multimodel_lipread/video/plots/{model_name}_loss_plot.png"
    plt.savefig(loss_path)
    print(f"[Saved] Loss plot → {loss_path}")
    plt.show()

    # ===========================
    # ACCURACY PLOT
    # ===========================
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc", marker="o")
    plt.plot(df["epoch"], df["test_acc"], label="Test Acc", marker="o")

    plt.title(f"{model_name} - Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    acc_path = f"/home/aswath/Projects/capstone/multimodel_lipread/video/plots/{model_name}_accuracy_plot.png"
    plt.savefig(acc_path)
    print(f"[Saved] Accuracy plot → {acc_path}")
    plt.show()


if __name__ == "__main__":
    model_name = "2dcnn_bilstm"   # change as needed
    plot_logs(model_name)