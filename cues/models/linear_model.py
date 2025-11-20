import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset, DatasetDict, ClassLabel
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm
from config.config import load_config
import numpy as np



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def get_constants(config, mode="env"):
    data_dir = config.get("old_description.input_dir")
    json_files = [
        f"lipreading_analysis_results_{mode}_aufgaben.json",
        f"lipreading_analysis_results_{mode}_dagegen.json",
        f"lipreading_analysis_results_{mode}_lieber.json",
        f"lipreading_analysis_results_{mode}_sein.json",
    ]

    MAX_FEATURES = 5000       # TF-IDF vocabulary size
    BATCH_SIZE = 32           # larger batch OK now
    SEED = 42
    NUM_EPOCHS = 8            # train longer (cheap model)
    LR = 1e-3                 # higher LR for small models

    return data_dir, json_files, MAX_FEATURES, BATCH_SIZE, SEED, NUM_EPOCHS, LR



# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_raw_dataset(data_dir, json_files):
    data_files = {"train": [f"{data_dir}/{fn}" for fn in json_files]}
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
# TF-IDF Vectorization
# ---------------------------------------------------------------------------
def vectorize_text(train_texts: List[str], test_texts: List[str], max_features: int):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # bigrams help classification
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    return X_train, X_test, vectorizer



# ---------------------------------------------------------------------------
# PyTorch Dataset Prep
# ---------------------------------------------------------------------------
def prepare_dataset(tokenized_train, tokenized_test, labels_train, labels_test):
    train_tensor = TensorDataset(
        torch.tensor(tokenized_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long)
    )
    test_tensor = TensorDataset(
        torch.tensor(tokenized_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.long)
    )
    return train_tensor, test_tensor



# ---------------------------------------------------------------------------
# Simple MLP Classifier
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )

    def forward(self, x):
        return self.net(x)



# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    return total_loss / len(loader), total_correct / total_samples



# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(model, train_loader, valid_loader, device, lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, valid_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}\n")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load config
    config_path = "./config/cues_config.yaml"
    config = load_config(config_path)

    data_dir, json_files, max_features, batch_size, seed, epochs, lr = get_constants(config)

    # Load dataset
    raw_ds = load_raw_dataset(data_dir, json_files)
    raw_ds, _ = encode_labels(raw_ds)

    # Split
    splits = raw_ds.train_test_split(test_size=0.1, seed=seed)
    train_ds, test_ds = splits["train"], splits["test"]

    # Vectorize descriptions
    X_train, X_test, vectorizer = vectorize_text(
        train_ds["description"],
        test_ds["description"],
        max_features,
    )

    # Prepare tensors
    train_tensor, test_tensor = prepare_dataset(X_train, X_test, train_ds["word"], test_ds["word"])

    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    # Build and train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim=X_train.shape[1], num_labels=len(set(train_ds["word"]))).to(device)

    train(model, train_loader, valid_loader, device, lr, epochs)

    # Save
    torch.save(model.state_dict(), "mlp_lipread_classifier.pt")
    print("Model saved!")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
