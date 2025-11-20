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
def get_constants(config, mode="env"):
    data_dir = config.get("old_description.input_dir")
    json_files = [
        f"lipreading_analysis_results_{mode}_aufgaben.json",
        f"lipreading_analysis_results_{mode}_dagegen.json",
        f"lipreading_analysis_results_{mode}_lieber.json",
        f"lipreading_analysis_results_{mode}_sein.json",
    ]

    MODEL_NAME = "bert-base-uncased"  # change to distilbert if needed
    MAX_LENGTH = 128
    BATCH_SIZE = 2   # reduced for 2GB GPU
    SEED = 42
    NUM_EPOCHS = 3
    LR = 2e-5
    WARMUP_STEPS = 0

    return data_dir, json_files, MODEL_NAME, MAX_LENGTH, BATCH_SIZE, SEED, NUM_EPOCHS, LR, WARMUP_STEPS


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_raw_dataset(data_dir, json_files) -> DatasetDict:
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
# Tokenization
# ---------------------------------------------------------------------------
def get_tokenizer(MODEL_NAME):
    return BertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_fn(example: Dict, tokenizer, MAX_LENGTH) -> Dict:
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
    return raw_ds.map(
        lambda x: tokenize_fn(x, tokenizer, MAX_LENGTH),
        batched=False,
        remove_columns=["word", "sequence_id", "description"],
    )


# ---------------------------------------------------------------------------
# Dataset Splitting & Formatting
# ---------------------------------------------------------------------------
def prepare_dataset(tokenized, SEED, MODEL_NAME):
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
    labels = list(set(example["labels"].item() for example in dataset["train"]))
    label2id = {label: i for i, label in enumerate(sorted(labels))}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label2id)
    return label2id, id2label, num_labels


# ---------------------------------------------------------------------------
# Model Setup (FP16)
# ---------------------------------------------------------------------------
def build_model(num_labels, label2id, id2label, device, MODEL_NAME):
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples
    return avg_lo_
