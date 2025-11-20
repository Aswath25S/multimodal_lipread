import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import (
    BertTokenizerFast,
    DataCollatorWithPadding,
)

# 1) CONFIGURATION
# ------------------------------------------------------------------------------
DATA_DIR = Path("./")  # adjust if your JSONs live elsewhere
JSON_FILES = [
    "lipreading_analysis_results_env_aufgaben.json",
    "lipreading_analysis_results_env_dagegen.json",
    "lipreading_analysis_results_env_lieber.json",
    "lipreading_analysis_results_env_sein.json",
]

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
SEED = 42

# 2) LOAD & CONCAT ALL JSONS AS A SINGLE HF DATASET
# ------------------------------------------------------------------------------
# build dict for load_dataset
data_files = {
    "train": [ str(DATA_DIR/fn) for fn in JSON_FILES ]
}

# each file is a list of records with fields: word, sequence_id, description
ds = load_dataset("json", data_files=data_files)
raw_ds = ds["train"]

# 3) LABEL‑ENCODE THE "word" FIELD
# ------------------------------------------------------------------------------
# find all unique tokens in `word`
unique_words = sorted(set(raw_ds["word"]))
# create a ClassLabel feature
word_feature = ClassLabel(names=unique_words)
# cast the column
raw_ds = raw_ds.cast_column("word", word_feature)

# 4) TOKENIZER SETUP
# ------------------------------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_fn(example: Dict) -> Dict:
    # tokenize the description; return input_ids + attention_mask
    toks = tokenizer(
        example["description"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "labels": example["word"],  # numerical label from ClassLabel 
    }

# 5) TOKENIZE & FORMAT FOR PYTORCH
# ------------------------------------------------------------------------------
# 5a) map tokenizer over the dataset
tokenized = raw_ds.map(
    tokenize_fn,
    batched=False,
    remove_columns=["word", "sequence_id", "description"],
)

# 5b) split into train/validation if you like
splits = tokenized.train_test_split(test_size=0.1, seed=SEED)
dataset = DatasetDict({
    "train": splits["train"],
    "validation": splits["test"],
})

# 5c) set PyTorch tensors and collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

for split in dataset.keys():
    dataset[split].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

# 6) CREATE DATALOADERS
# ------------------------------------------------------------------------------
train_loader = DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)
valid_loader = DataLoader(
    dataset["validation"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)

# 7) SANITY‐CHECK ONE BATCH
# ------------------------------------------------------------------------------
batch = next(iter(train_loader))
print("batch keys:", batch.keys())
print("input_ids.shape:", batch["input_ids"].shape)
print("attention_mask.shape:", batch["attention_mask"].shape)
print("labels.shape:", batch["labels"].shape)


# In[17]:


# Extract unique labels from the training data
labels = list(set(example['labels'].item() for example in dataset["train"]))
label2id = {label: i for i, label in enumerate(sorted(labels))}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label2id)


# In[19]:


import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


# 1) HYPERPARAMETERS
NUM_EPOCHS   = 3
LR           = 2e-5
WARMUP_STEPS = 0
MODEL_NAME   = "bert-base-uncased"

# 2) DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) MODEL
#    Assume `num_labels` is the number of distinct words (e.g. len(unique_words))
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)
model.to(device)

# 4) OPTIMIZER & SCHEDULER
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)

# 5) EVAL FUNCTION
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss  = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct  += (preds == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy

# 6) TRAIN/VALID LOOP
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} — Training"):
        optimizer.zero_grad()
        
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    val_loss, val_acc = evaluate(valid_loader)

    print(f"\n=== Epoch {epoch+1} Summary ===")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Valid Loss: {val_loss:.4f}  |  Valid Acc: {val_acc:.4f}\n")

# 7) SAVE YOUR MODEL (optional)
model.save_pretrained("./bert-lipread-classifier")


# In[ ]:




