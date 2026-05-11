#!/usr/bin/env python3

import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import pyarrow.parquet as pq

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.amp import autocast, GradScaler

# -----------------------------
# DEBUG ENV (set before init_process_group)
# -----------------------------
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

# -----------------------------
# CONFIG
# -----------------------------
PARQUET_DIR = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
LABEL_FILE = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3

# logging config
LOG_EVERY = 10          # log every N batches
SLOW_BATCH_SEC = 120    # warn if a batch takes > 120s

# -----------------------------
# DDP SETUP
# -----------------------------
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# -----------------------------
# LOAD LABELS (FIXED)
# -----------------------------
labels_df = pd.read_csv(LABEL_FILE)
labels_df["binary_label"] = (labels_df["summary_score"] >= 15).astype(int)
label_dict = dict(zip(labels_df["subject_sp_id"], labels_df["binary_label"]))

# -----------------------------
# DATASET
# -----------------------------
class VariantDataset(Dataset):
    def __init__(self, files, label_dict):
        self.files = files
        self.label_dict = label_dict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]

        table = pq.read_table(f, columns=["embedding"])
        df = table.to_pandas()

        emb = np.stack(df["embedding"].values)
        sample_id = os.path.basename(f).split(".")[0]

        label = self.label_dict.get(sample_id, None)
        if label is None:
            raise ValueError(f"Missing label for {sample_id}")

        return {"emb": emb, "label": label, "file": f}

# -----------------------------
# COLLATE
# -----------------------------
def collate_fn(batch):
    embs = [torch.tensor(item["emb"], dtype=torch.float32) for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    lengths = [e.shape[0] for e in embs]
    max_len = max(lengths)

    embed_dim = embs[0].shape[1]

    padded = torch.zeros(len(batch), max_len, embed_dim)
    mask = torch.zeros(len(batch), max_len)

    for i, e in enumerate(embs):
        padded[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = 1

    return padded, mask, labels

# -----------------------------
# MODEL (FIXED: NO SIGMOID)
# -----------------------------
class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores[mask == 0] = -1e4
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return pooled

class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pool = AttentionPooling(input_dim)
        self.classifier = torch.nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        pooled = self.pool(x, mask)
        return self.classifier(pooled).squeeze()  # logits

model = Model(1280).to(device)
model = DDP(model, device_ids=[local_rank])

# -----------------------------
# LOSS (FIXED)
# -----------------------------
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scaler = GradScaler("cuda")

# -----------------------------
# LOAD FILES
# -----------------------------
all_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))

valid_files = []
for f in all_files:
    sample_id = os.path.basename(f).split(".")[0]
    if sample_id in label_dict:
        valid_files.append(f)

train_files, test_files = train_test_split(
    valid_files, test_size=0.2, random_state=42
)

train_dataset = VariantDataset(train_files, label_dict)
test_dataset = VariantDataset(test_files, label_dict)

train_sampler = DistributedSampler(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# -----------------------------
# TRAINING
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_sampler.set_epoch(epoch)

    total_loss = 0.0
    epoch_start = time.time()

    for step, (x, mask, y) in enumerate(train_loader, start=1):
        step_start = time.time()

        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            out = model(x, mask)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        step_time = time.time() - step_start
        if step % LOG_EVERY == 0:
            print(
                f"[rank {local_rank}] "
                f"epoch {epoch+1}/{EPOCHS} step {step}/{len(train_loader)} "
                f"loss={loss.item():.4f} step_time={step_time:.2f}s"
            )
        if step_time > SLOW_BATCH_SEC:
            print(
                f"[rank {local_rank}] WARNING: slow batch "
                f"epoch {epoch+1} step {step} time={step_time:.2f}s"
            )

    if local_rank == 0:
        print(
            f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, "
            f"epoch_time={time.time() - epoch_start:.2f}s"
        )

# -----------------------------
# EVALUATION (FIXED)
# -----------------------------
if local_rank == 0:
    model.eval()

    probs = []
    preds = []
    y_true = []

    with torch.no_grad():
        for x, mask, y in test_loader:
            x = x.to(device)
            mask = mask.to(device)

            out = model(x, mask)

            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob > 0.5).astype(int)

            probs.extend(prob)
            preds.extend(pred)
            y_true.extend(y.numpy())

    y_true = np.array(y_true)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    roc = roc_auc_score(y_true, probs)
    f1_binary = f1_score(y_true, preds, average='binary')

    print("\n--- FINAL EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Binary F1: {f1_binary:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    print("\nConfusion Matrix:")
    print(f"TP (ASD correctly predicted): {tp}")
    print(f"FP (Non-ASD predicted as ASD): {fp}")
    print(f"TN (Non-ASD correctly predicted): {tn}")
    print(f"FN (ASD missed): {fn}")

# -----------------------------
# CLEAN EXIT (DDP FIX)
# -----------------------------
dist.destroy_process_group()