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
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/cues')
from config.config import load_config

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

    # all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2, all-distilroberta-v1
    EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    BATCH_SIZE = 16
    SEED = 42
    NUM_EPOCHS = 12
    LR = 1e-3
    WARMUP_PROPORTION = 0.1

    return data_dir, base_path, json_files, EMB_MODEL_NAME, BATCH_SIZE, SEED, NUM_EPOCHS, LR, WARMUP_PROPORTION

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
# Tokenizer + Embedding
# ---------------------------------------------------------------------------
def get_tokenizer_and_model(emb_model_name):
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
    model = AutoModel.from_pretrained(emb_model_name)
    return tokenizer, model

def embed_sentences_tokenwise(tokenizer, model, sentences, device, max_length=32):
    model.to(device)
    all_embeddings = []

    for sent in tqdm(sentences, desc="Embedding sentences"):
        encoded = tokenizer(sent, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeds = outputs.last_hidden_state.squeeze(0)
        all_embeddings.append(token_embeds.cpu().numpy())

    return np.stack(all_embeddings)

# ---------------------------------------------------------------------------
# CNN + BiLSTM + Multi-head Self-Attention Classifier
# ---------------------------------------------------------------------------
class CNNBiLSTMAttn(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, kernel_sizes=[2,3,4], n_filters=64, n_heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k) for k in kernel_sizes])
        self.lstm = nn.LSTM(n_filters*len(kernel_sizes), hidden_dim, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=n_heads, batch_first=True)
        self.output_dense = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        x = x.transpose(1,2)
        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        pooled_outs = [torch.max(out, dim=2)[0] for out in conv_outs]
        cnn_out = torch.cat(pooled_outs, dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(cnn_out)
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.mean(dim=1)
        logits = self.output_dense(attn_out)
        return logits

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
# Training Loop with Logging
# ---------------------------------------------------------------------------
def train(model, train_loader, valid_loader, device, lr, epochs, metrics_path, model_path,
          class_weights, warmup_steps, model_name="minilm_cnn_bilstm_attn"):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader)*epochs
    )

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
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            pbar.set_postfix({"loss": total_loss/len(train_loader),
                              "acc": 100.0*total_correct/total_samples})

        train_loss = total_loss/len(train_loader)
        train_acc = 100.0*total_correct/total_samples
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

    data_dir, base_path, json_files, EMB_MODEL, BATCH_SIZE, SEED, EPOCHS, LR, WARMUP_PROP = get_constants(config)
    metrics_path = f"{base_path}/metrics"
    models_path = f"{base_path}/models_trained"

    raw_ds = load_raw_dataset(data_dir, json_files)
    raw_ds, unique_words = encode_labels(raw_ds)
    splits = raw_ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds, test_ds = splits["train"], splits["test"]

    tokenizer, emb_model = get_tokenizer_and_model(EMB_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model.to(device)

    print("Embedding training set (token-level)...")
    X_train = embed_sentences_tokenwise(tokenizer, emb_model, train_ds["description"], device)
    print("Embedding validation set (token-level)...")
    X_test = embed_sentences_tokenwise(tokenizer, emb_model, test_ds["description"], device)

    y_train = np.array(train_ds["word"])
    y_test = np.array(test_ds["word"])

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=False)

    embed_dim = X_train.shape[2]
    model = CNNBiLSTMAttn(embed_dim=embed_dim, hidden_dim=128, num_labels=len(unique_words))
    model.to(device)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_PROP * total_steps)

    train(model, train_loader, valid_loader, device, LR, EPOCHS, metrics_path, models_path,
          class_weights, warmup_steps, model_name="minilm_cnn_bilstm_attn")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
