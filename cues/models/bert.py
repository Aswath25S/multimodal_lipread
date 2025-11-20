import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import (
    BertTokenizerFast,
    DataCollatorWithPadding,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import numpy as np
from tqdm.auto import tqdm
from config.config import load_config


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
def get_constants(config, mode = "env"):
    data_dir = config.get("old_description.input_dir")
    json_files = [
        f"lipreading_analysis_results_{mode}_aufgaben.json",
        f"lipreading_analysis_results_{mode}_dagegen.json",
        f"lipreading_analysis_results_{mode}_lieber.json",
        f"lipreading_analysis_results_{mode}_sein.json",
    ]

    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    SEED = 42
    NUM_EPOCHS = 3
    LR = 2e-5
    WARMUP_STEPS = 0

    return data_dir, json_files, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, SEED, NUM_EPOCHS, LR, WARMUP_STEPS


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_raw_dataset(data_dir, json_files) -> DatasetDict:
    """
    Load the JSON files into a single HuggingFace dataset.
    Each record contains: word, sequence_id, description.
    """
    data_files = {"train": [f"{data_dir}/{fn}" for fn in json_files]}
    dataset = load_dataset("json", data_files=data_files)
    return dataset["train"]


# ---------------------------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------------------------
def encode_labels(raw_ds):
    """
    Convert the 'word' field into numerical class IDs using ClassLabel.
    """
    unique_words = sorted(set(raw_ds["word"]))
    word_feature = ClassLabel(names=unique_words)
    encoded = raw_ds.cast_column("word", word_feature)
    return encoded, unique_words


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def get_tokenizer(MODEL_NAME):
    """Load the BERT tokenizer."""
    return BertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_fn(example: Dict, tokenizer, MAX_LENGTH) -> Dict:
    """
    Convert textual descriptions into token IDs and masks.
    Also attach the numerical label associated with each record.
    """
    toks = tokenizer(
        example["description"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "labels": example["word"],
    }


def tokenize_dataset(raw_ds, tokenizer, MAX_LENGTH):
    """
    Apply tokenization to the entire dataset and drop unused columns.
    """
    return raw_ds.map(
        lambda x: tokenize_fn(x, tokenizer, MAX_LENGTH),
        batched=False,
        remove_columns=["word", "sequence_id", "description"],
    )


# ---------------------------------------------------------------------------
# Dataset Splitting & Formatting
# ---------------------------------------------------------------------------
def prepare_dataset(tokenized, SEED, MODEL_NAME):
    """
    Split into training/validation and make datasets PyTorch-compatible.
    """
    splits = tokenized.train_test_split(test_size=0.1, seed=SEED)

    dataset = DatasetDict({
        "train": splits["train"],
        "validation": splits["test"],
    })

    tokenizer = get_tokenizer(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    for split in dataset.keys():
        dataset[split].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    return dataset, data_collator


# ---------------------------------------------------------------------------
# DataLoader Creation
# ---------------------------------------------------------------------------
def create_dataloaders(dataset, collator, BATCH_SIZE):
    """
    Build PyTorch dataloaders for training and validation.
    """
    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        dataset["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, valid_loader


# ---------------------------------------------------------------------------
# Label Mapping
# ---------------------------------------------------------------------------
def extract_label_maps(dataset):
    """
    Create id→label and label→id dictionaries from dataset labels.
    """
    labels = list(set(example["labels"].item() for example in dataset["train"]))
    label2id = {label: i for i, label in enumerate(sorted(labels))}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label2id)
    return label2id, id2label, num_labels


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------
def build_model(num_labels, label2id, id2label, device, MODEL_NAME):
    """
    Load a BERT classifier for the given number of output classes.
    """
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    """
    Compute average loss and accuracy over a dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(model, train_loader, valid_loader, device, LR, NUM_EPOCHS, WARMUP_STEPS):
    """
    Full training+validation loop using AdamW and linear LR scheduler.
    """
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} — Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, valid_loader, device)

        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}\n")

    model.save_pretrained("./bert-lipread-classifier")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config_path = "./config/cues_config.yaml"
    config = load_config(config_path)

    data_dir, json_files, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, SEED, NUM_EPOCHS, LR, WARMUP_STEPS = get_constants(config, "env")

    raw_ds = load_raw_dataset(data_dir, json_files)
    raw_ds, _ = encode_labels(raw_ds)

    tokenizer = get_tokenizer(MODEL_NAME)
    tokenized = tokenize_dataset(raw_ds, tokenizer, MAX_LENGTH)

    dataset, collator = prepare_dataset(tokenized, SEED, MODEL_NAME)

    train_loader, valid_loader = create_dataloaders(dataset, collator, BATCH_SIZE)

    label2id, id2label, num_labels = extract_label_maps(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_labels, label2id, id2label, device, MODEL_NAME)

    train(model, train_loader, valid_loader, device, LR, NUM_EPOCHS, WARMUP_STEPS)
    return


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
