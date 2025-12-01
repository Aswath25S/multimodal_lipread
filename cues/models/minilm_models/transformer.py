import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, ClassLabel
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.utils.class_weight import compute_class_weight

sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/cues')
from config.config import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def get_constants(config, mode="env"):
    data_dir = config.get("old_description.input_dir")
    base_path = config.get("main.base_path")

    json_files = [
        f"lipreading_analysis_results_{mode}_aufgaben.json",
        f"lipreading_analysis_results_{mode}_dagegen.json",
        f"lipreading_analysis_results_{mode}_lieber.json",
        f"lipreading_analysis_results_{mode}_sein.json",
    ]

    # Multiple Sentence Transformers for ensemble embeddings
    EMB_MODEL_NAMES = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    BATCH_SIZE = 8
    SEED = 42
    NUM_EPOCHS = 30
    LR = 1e-3

    return data_dir, base_path, json_files, EMB_MODEL_NAMES, BATCH_SIZE, SEED, NUM_EPOCHS, LR

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
# Multi-transformer Embedding
# ---------------------------------------------------------------------------
def embed_with_multiple_models(model_names, sentences):
    all_embeddings = []
    for name in model_names:
        print(f"Embedding with {name}...")
        model = SentenceTransformer(name)
        emb = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
        all_embeddings.append(emb)
    return np.concatenate(all_embeddings, axis=1)

# ---------------------------------------------------------------------------
# Multi-layer Transformer-style Classifier
# ---------------------------------------------------------------------------
class MultiAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_labels, hidden_dim=512, n_heads=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dense = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_dense = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        x = self.input_dense(x)
        x = x.unsqueeze(1)
        for attn in self.attention_layers:
            attn_out, _ = attn(x, x, x)
            x = attn_out + x  # residual
        x = x.squeeze(1)
        logits = self.output_dense(x)
        return logits

# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------
def init_log_files(model_name, metrics_path):
    os.makedirs(metrics_path, exist_ok=True)
    csv_path = f"{metrics_path}/{model_name}_training_log.csv"
    txt_path = f"{metrics_path}/{model_name}_training_log.txt"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write("Training Log\n\n")

def log_to_files(model_name, metrics_path, epoch, train_loss, train_acc, val_loss, val_acc):
    csv_path = f"{metrics_path}/{model_name}_training_log.csv"
    txt_path = f"{metrics_path}/{model_name}_training_log.txt"
    with open(csv_path, "a") as f:
        f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n")
    with open(txt_path, "a") as f:
        f.write(
            f"Epoch {epoch}\n"
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n\n"
        )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion, device, desc="Validating"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss/len(loader), 100.0*correct/total

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(model, train_loader, valid_loader, device, lr, epochs, metrics_path, model_path, class_weights, model_name="multi_attention"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    os.makedirs(model_path, exist_ok=True)
    init_log_files(model_name, metrics_path)
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

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
            pbar.set_postfix({"loss": total_loss/len(train_loader),
                              "acc": 100.0*total_correct/total_samples})

        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * total_correct / total_samples
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        log_to_files(model_name, metrics_path, epoch, train_loss, train_acc, val_loss, val_acc)

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
# Main
# ---------------------------------------------------------------------------
def main():
    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/cues/config/cues_config.yaml"
    config = load_config(config_path)

    data_dir, base_path, json_files, EMB_MODELS, BATCH_SIZE, SEED, EPOCHS, LR = get_constants(config)
    metrics_path = f"{base_path}/metrics"
    models_path = f"{base_path}/models_trained"

    raw_ds = load_raw_dataset(data_dir, json_files)
    raw_ds, unique_words = encode_labels(raw_ds)

    splits = raw_ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds, test_ds = splits["train"], splits["test"]

    print("Embedding training set with multiple transformers...")
    X_train = embed_with_multiple_models(EMB_MODELS, train_ds["description"])
    print("Embedding validation set with multiple transformers...")
    X_test = embed_with_multiple_models(EMB_MODELS, test_ds["description"])

    y_train = np.array(train_ds["word"])
    y_test = np.array(test_ds["word"])

    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train),
        dtype=torch.float32
    )

    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=False)

    embed_dim = X_train.shape[1]
    model = MultiAttentionClassifier(embed_dim=embed_dim, num_labels=len(unique_words),
                                     hidden_dim=512, n_heads=8, num_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, train_loader, valid_loader, device, LR, EPOCHS, metrics_path, models_path, class_weights, model_name="multi_attention")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
