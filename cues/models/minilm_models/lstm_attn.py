import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, ClassLabel
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/cues')
from config.config import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def get_constants(config, mode="emotion"):
    data_dir = config.get("old_description.input_dir")
    base_path = config.get("main.base_path")

    json_files = [
        f"lipreading_analysis_results_{mode}_aufgaben.json",
        f"lipreading_analysis_results_{mode}_dagegen.json",
        f"lipreading_analysis_results_{mode}_lieber.json",
        f"lipreading_analysis_results_{mode}_sein.json",
    ]

    # all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2, all-distilroberta-v1
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    BATCH_SIZE = 8
    SEED = 42
    NUM_EPOCHS = 30
    LR = 1e-3

    return data_dir, base_path, json_files, EMBEDDING_MODEL, BATCH_SIZE, SEED, NUM_EPOCHS, LR


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_raw_dataset(data_dir, json_files):
    data_files = {"train": [os.path.join(data_dir, fn) for fn in json_files]}
    dataset = load_dataset("json", data_files=data_files)
    return dataset["train"]


# ---------------------------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------------------------
def encode_labels(raw_ds):
    unique_words = sorted(set(raw_ds["word"]))
    word_feature = ClassLabel(names=unique_words)
    encoded = raw_ds.cast_column("word", word_feature)
    return encoded, unique_words


# ---------------------------------------------------------------------------
# Embedding Function
# ---------------------------------------------------------------------------
def embed_sentences(model, sentences):
    return model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=True
    )


# ---------------------------------------------------------------------------
# Attention + BiLSTM Classifier
# ---------------------------------------------------------------------------
class AttentionLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels):
        super().__init__()

        # Dense projection before LSTM
        self.input_dense = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Output layers
        self.output_dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        # x shape: (batch, embed_dim)
        x = self.input_dense(x)          # (batch, 256)
        x = x.unsqueeze(1)               # (batch, seq_len=1, 256)

        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden_dim*2)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)

        # Weighted sum over LSTM outputs
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)         # (batch, hidden_dim*2)

        logits = self.output_dense(attn_output)
        return logits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion, device, desc="Validating"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def init_log_files(model_name, metrics_path):
    os.makedirs(metrics_path, exist_ok=True)
    csv_path = f"{metrics_path}/{model_name}_training_log.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")


def log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc):
    metrics_path = "/home/aswath/Projects/capstone/multimodel_lipread/cues/metrics"
    with open(f"{metrics_path}/{model_name}_training_log.csv", "a") as f:
        f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n")
    with open(f"{metrics_path}/{model_name}_training_log.txt", "a") as f:
        f.write(
            f"Epoch {epoch}\n"
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n\n"
        )


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(model, train_loader, valid_loader, device, lr, epochs, metrics_path, model_path, model_name="minilm_lstm_attn"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(model_path, exist_ok=True)
    init_log_files(model_name, metrics_path)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            pbar.set_postfix({
                "loss": total_loss / len(train_loader),
                "acc": 100.0 * total_correct / total_samples
            })

        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * total_correct / total_samples

        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")

        log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, f"{model_path}/{model_name}.pth")

            print(f"✔ Saved new best model (Val Acc: {val_acc:.2f}%)\n")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/cues/config/cues_config.yaml"
    config = load_config(config_path)

    data_dir, base_path, json_files, EMB_MODEL, BATCH_SIZE, SEED, EPOCHS, LR = get_constants(config)
    metrics_path = f"{base_path}/metrics"
    models_path = f"{base_path}/models_trained"

    raw_ds = load_raw_dataset(data_dir, json_files)
    raw_ds, unique_words = encode_labels(raw_ds)

    splits = raw_ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds, test_ds = splits["train"], splits["test"]

    embedding_model = SentenceTransformer(EMB_MODEL)

    print("\nEmbedding training set...")
    X_train = embed_sentences(embedding_model, train_ds["description"])
    print("Embedding validation set...")
    X_test = embed_sentences(embedding_model, test_ds["description"])

    y_train = np.array(train_ds["word"])
    y_test = np.array(test_ds["word"])

    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=False)

    embed_dim = X_train.shape[1]
    model = AttentionLSTMClassifier(embed_dim=embed_dim, hidden_dim=128, num_labels=len(unique_words))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, train_loader, valid_loader, device, LR, EPOCHS, metrics_path, models_path, model_name="minilm_lstm_attn")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
