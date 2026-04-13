"""
TransformerMIL vs ABMIL Comparison Classifier
for Genomic Variant Embeddings → Social Interaction Classification

DISTRIBUTED TRAINING with DistributedDataParallel:
  Run with: torchrun --nproc_per_node=4 Transformer_ABMIL_Classifier_modified.py
"""

import os
import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    balanced_accuracy_score, roc_curve
)
import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DDP SETUP
# ─────────────────────────────────────────────
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=86400))
local_rank = int(os.environ["LOCAL_RANK"])
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def is_main_process():
    return rank == 0


def log(msg):
    """Print only from main process"""
    if is_main_process():
        print(msg)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PARQUET_DIR = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
CSV_PATH = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"
SCORE_THRESH = 15
EMBEDDING_COL = "embedding"

# Variant caps per model
MAX_VARIANTS_ABMIL = None  # All ~46k variants
MAX_VARIANTS_TRANSFORMER = 4096  # Capped for GPU memory

RANDOM_SEED = 42
N_FOLDS = 5

# Shared hyper-params
EMBED_DIM = None  # Auto-detected
HIDDEN_DIM = 256
DROPOUT = 0.1
EPOCHS = 60
BATCH_SIZE = 32  # Per GPU batch size
PATIENCE = 15
N_WORKERS = 2     # 2 workers per rank — reduces pinned memory buffer accumulation
                  # 4 ranks × 2 workers = 8 total worker processes
LR = 3e-4

# ABMIL-specific
ATTN_DIM = 128

# Transformer-specific
N_HEADS = 8
N_LAYERS = 4
FFN_DIM = 512
MAX_SEQ_LEN = MAX_VARIANTS_TRANSFORMER + 1

# Mixed precision
USE_AMP = True

LAZY_LOAD = True   # ✅ MUST be True for DDP — each rank reads only its own batches
                   # False would load ALL ~83GB into EVERY rank = 332GB total RAM

if is_main_process():
    log(f"Device: {device}")
    log(f"Rank: {rank}/{world_size}")
    log(f"Local Rank: {local_rank}")


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def extract_subject_id(filename: str) -> str:
    m = re.match(r"(SP\d+)", os.path.basename(filename))
    return m.group(1) if m else None


def load_labels(csv_path: str, threshold: int) -> dict:
    df = pd.read_csv(csv_path)
    df["label"] = (df["summary_score"] > threshold).astype(int)
    return dict(zip(df["subject_sp_id"].astype(str), df["label"]))


def load_subject_embeddings(parquet_path: str, max_variants=None) -> np.ndarray:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=[EMBEDDING_COL],memory_map=True)
    df = table.to_pandas()
    embs = df[EMBEDDING_COL].tolist()
    arr = np.array([np.array(e, dtype=np.float32) for e in embs])
    if max_variants and len(arr) > max_variants:
        idx = np.random.choice(len(arr), max_variants, replace=False)
        arr = arr[idx]
    return arr


def get_embedding_array(record, max_variants=None):
    """
    Always loads from disk — record[0] is always a file path.
    Each DataLoader worker reads only the files assigned to its rank's batch.
    This is the key to keeping RAM low in DDP.
    """
    return load_subject_embeddings(record[0], max_variants=max_variants)


def build_dataset(parquet_dir, label_dict):
    """
    Builds index of (path, label, sid) tuples.
    Always stores paths only — actual embeddings loaded lazily per batch.
    This keeps RAM usage minimal regardless of dataset size.
    """
    records = []
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    if is_main_process():
        log(f"Found {len(files)} parquet files")

    missing = 0
    for fname in files:
        sid = extract_subject_id(fname)
        if sid is None or sid not in label_dict:
            missing += 1
            continue
        path = os.path.join(parquet_dir, fname)
        records.append((path, label_dict[sid], sid))   # always store path only

    if is_main_process():
        log(f"Indexed {len(records)} subjects | skipped {missing}")
        labels = [r[1] for r in records]
        log(f"Class distribution → 0: {labels.count(0)} | 1: {labels.count(1)}")

    return records


# ─────────────────────────────────────────────
# 2. DATASETS
# ─────────────────────────────────────────────

class BagDataset(Dataset):
    """Variable-length bags for ABMIL"""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        emb = get_embedding_array(self.records[idx], max_variants=MAX_VARIANTS_ABMIL)
        label = self.records[idx][1]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class PaddedBagDataset(Dataset):
    """Padded bags for Transformer"""

    def __init__(self, records, max_len, embed_dim):
        self.records = records
        self.max_len = max_len
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        emb = get_embedding_array(self.records[idx],
                                  max_variants=MAX_VARIANTS_TRANSFORMER)
        label = self.records[idx][1]
        N = len(emb)

        if N >= self.max_len:
            emb_out = emb[:self.max_len]
            pad_mask = torch.zeros(self.max_len, dtype=torch.bool)
        else:
            pad_len = self.max_len - N
            padding = np.zeros((pad_len, self.embed_dim), dtype=np.float32)
            emb_out = np.concatenate([emb, padding], axis=0)
            pad_mask = torch.tensor(
                [False] * N + [True] * pad_len, dtype=torch.bool
            )

        return (
            torch.tensor(emb_out, dtype=torch.float32),
            pad_mask,
            torch.tensor(label, dtype=torch.long)
        )


def collate_bags(batch):
    bags, labels = zip(*batch)
    return list(bags), torch.stack(labels)


def collate_padded(batch):
    embs, masks, labels = zip(*batch)
    return torch.stack(embs), torch.stack(masks), torch.stack(labels)


# ─────────────────────────────────────────────
# 3a. ABMIL MODEL
# ─────────────────────────────────────────────

class AttentionMIL(nn.Module):
    """Gated Attention MIL — vectorized for GPU efficiency"""

    def __init__(self, embed_dim, hidden_dim=256, attn_dim=128,
                 dropout=0.1, n_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, bags, return_attention=False):
        # ── move all bags to device and pad to same length ──
        bags = [b.to(device) for b in bags]       # ✅ explicit GPU move
        max_len = max(len(b) for b in bags)

        padded_bags, masks = [], []
        for bag in bags:
            pad_len = max_len - len(bag)
            if pad_len > 0:
                bag = torch.cat([bag,
                    torch.zeros(pad_len, bag.shape[1], device=device)])
            mask = torch.zeros(max_len, device=device)
            mask[:len(bag) - pad_len] = 1.0
            padded_bags.append(bag)
            masks.append(mask)

        bags_tensor  = torch.stack(padded_bags)   # (B, max_len, embed_dim)
        masks_tensor = torch.stack(masks)          # (B, max_len)

        H   = self.encoder(bags_tensor)            # (B, max_len, hidden_dim)
        A_V = torch.tanh(self.attn_V(H))
        A_U = torch.sigmoid(self.attn_U(H))
        A   = self.attn_w(A_V * A_U)              # (B, max_len, 1)

        # mask padding positions before softmax
        A = A + (1 - masks_tensor.unsqueeze(-1)) * (-1e9)
        A = F.softmax(A, dim=1)                    # (B, max_len, 1)

        z      = (A * H).sum(dim=1)               # (B, hidden_dim)
        logits = self.classifier(z)
        return logits

# ─────────────────────────────────────────────
# 3b. TRANSFORMER MIL MODEL
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=600, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerMIL(nn.Module):
    """Transformer Encoder for MIL"""

    def __init__(self, embed_dim, d_model=256, n_heads=8, n_layers=4,
                 ffn_dim=512, max_seq_len=513, dropout=0.1, n_classes=2):
        super().__init__()

        assert d_model % n_heads == 0

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x, pad_mask=None):
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if pad_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_out = x[:, 0, :]

        return self.classifier(cls_out)


# ─────────────────────────────────────────────
# 4. TRAINING UTILITIES
# ─────────────────────────────────────────────

def train_epoch_abmil(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for bags, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad()

        with autocast("cuda" if USE_AMP else "cpu"):
            logits = model(bags)
            loss = criterion(logits, labels)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                        # ✅ unscale before clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # ✅ clip gradients
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_epoch_transformer(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for embs, masks, labels in loader:
        embs, masks, labels = embs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast("cuda" if USE_AMP else "cpu"):
            logits = model(embs, masks)
            loss = criterion(logits, labels)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                         # ✅ unscale before clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ✅ clip gradients
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_abmil(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for bags, labels in loader:
        bags   = [b.to(device) for b in bags]     # ✅ move bags to GPU
        labels = labels.to(device)                 # ✅ move labels to GPU
        logits = model(bags)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())    # ✅ .cpu() before .numpy()

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def eval_transformer(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for embs, masks, labels in loader:
        embs, masks, labels = embs.to(device), masks.to(device), labels.to(device)  # ✅
        logits = model(embs, masks)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())    # ✅ .cpu() before .numpy()

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────
# 5. CV RUNNER
# ────────────────────���────────────────────────

def run_cv(model_name, records, embed_dim, n_splits=N_FOLDS):
    """Stratified K-fold CV"""
    log("\n" + "=" * 60)
    log(f"{model_name.upper()} — {n_splits}-Fold CV")
    log("=" * 60)

    indices = np.arange(len(records))
    labels = np.array([r[1] for r in records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_aucs, fold_baccs = [], []
    fold_roc_data = []
    best_model_state, best_auc = None, 0.0

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, labels)):
        torch.cuda.empty_cache()
        log(f"\n── Fold {fold + 1}/{n_splits} ──")
        tr_records = [records[i] for i in tr_idx]
        te_records = [records[i] for i in te_idx]

        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        scaler = GradScaler("cuda") if USE_AMP else None

        # ── Build model + dataloaders ──
        if model_name == "abmil":
            model = AttentionMIL(
                embed_dim=embed_dim, hidden_dim=HIDDEN_DIM,
                attn_dim=ATTN_DIM, dropout=DROPOUT
            ).to(device)

            # Wrap with DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,broadcast_buffers=True)
            tr_dataset = BagDataset(tr_records)
            # DistributedSampler for training
            tr_sampler = DistributedSampler(
                tr_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=RANDOM_SEED
            )
            tr_loader = DataLoader(
                tr_dataset,
                batch_size=BATCH_SIZE,
                sampler=tr_sampler,
                collate_fn=collate_bags,
                num_workers=N_WORKERS,
                pin_memory=True
            )

            te_dataset = BagDataset(te_records)
            # DistributedSampler for testing
            te_sampler = DistributedSampler(
                te_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=RANDOM_SEED
            )
            te_loader = DataLoader(
                te_dataset,
                batch_size=BATCH_SIZE,
                sampler=te_sampler,
                collate_fn=collate_bags,
                num_workers=N_WORKERS,
                pin_memory=True
            )
            train_fn = train_epoch_abmil
            eval_fn = eval_abmil

        else:  # transformer
            d_model = HIDDEN_DIM
            while d_model % N_HEADS != 0:
                d_model += 1

            model = TransformerMIL(
                embed_dim=embed_dim, d_model=d_model,
                n_heads=N_HEADS, n_layers=N_LAYERS,
                ffn_dim=FFN_DIM, max_seq_len=MAX_SEQ_LEN,
                dropout=DROPOUT
            ).to(device)

            # Wrap with DDP
            model = DDP(model, device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=False)

            # ✅ create dataset ONCE, reuse for both sampler and loader
            tr_dataset = PaddedBagDataset(tr_records, MAX_VARIANTS_TRANSFORMER, embed_dim)
            tr_sampler = DistributedSampler(
                tr_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=RANDOM_SEED
            )
            tr_loader = DataLoader(
                tr_dataset,  # ✅ reuse same object
                batch_size=BATCH_SIZE,
                sampler=tr_sampler,
                collate_fn=collate_padded,
                num_workers=N_WORKERS,
                pin_memory=True
            )

            # ✅ same fix for test set
            te_dataset = PaddedBagDataset(te_records, MAX_VARIANTS_TRANSFORMER, embed_dim)
            te_sampler = DistributedSampler(
                te_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=RANDOM_SEED
            )
            te_loader = DataLoader(
                te_dataset,  # ✅ reuse same object
                batch_size=BATCH_SIZE,
                sampler=te_sampler,
                collate_fn=collate_padded,
                num_workers=N_WORKERS,
                pin_memory=True
            )
            train_fn = train_epoch_transformer
            eval_fn = eval_transformer

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Model params: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_auc, patience_cnt, fold_state = 0.0, 0, None

        for epoch in range(1, EPOCHS + 1):
            # Set epoch for sampler (important for shuffle)
            tr_sampler.set_epoch(epoch)

            tr_loss, tr_acc = train_fn(model, tr_loader, optimizer, criterion, scaler)
            scheduler.step()
            y_true, y_pred, y_prob = eval_fn(model, te_loader)

            if is_main_process() and y_true is not None:
                val_auc = roc_auc_score(y_true, y_prob)
                val_bacc = balanced_accuracy_score(y_true, y_pred)

                if epoch % 10 == 0 or epoch == 1:
                    log(f"  Ep {epoch:3d} | loss={tr_loss:.4f} acc={tr_acc:.3f} "
                        f"| AUC={val_auc:.3f} BalAcc={val_bacc:.3f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_cnt = 0
                    # Access underlying model in DDP
                    fold_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        log(f"  Early stop at epoch {epoch}")
                        break

            # Sync barrier
            dist.barrier()

        if fold_state is not None and is_main_process():
            model.module.load_state_dict({k: v.to(device) for k, v in fold_state.items()})

        dist.barrier()

        y_true, y_pred, y_prob = eval_fn(model, te_loader)

        if is_main_process() and y_true is not None:
            auc = roc_auc_score(y_true, y_prob)
            bacc = balanced_accuracy_score(y_true, y_pred)
            fold_aucs.append(auc)
            fold_baccs.append(bacc)

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fold_roc_data.append((fpr, tpr, auc))

            log(f"\n  Fold {fold + 1} final → AUC={auc:.3f}  BalAcc={bacc:.3f}")
            log(classification_report(y_true, y_pred,
                                      target_names=["Typical (≤15)", "Atypical (>15)"]))

            if auc > best_auc:
                best_auc = auc
                best_model_state = fold_state

    if is_main_process():
        mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
        mean_bacc = np.mean(fold_baccs) if fold_baccs else 0.0
        log("\n" + "=" * 60)
        log(f"{model_name.upper()} SUMMARY")
        log(f"  Mean AUC   : {mean_auc:.3f} ± {np.std(fold_aucs) if fold_aucs else 0:.3f}")
        log(f"  Mean BalAcc: {mean_bacc:.3f} ± {np.std(fold_baccs) if fold_baccs else 0:.3f}")
        log("=" * 60)

        return best_model_state, fold_roc_data, mean_auc, mean_bacc
    else:
        return None, None, 0.0, 0.0


# ─────────────────────────────────────────────
# 6. COMPARISON ROC PLOT
# ─────────────────────────────────────────────

def plot_comparison_roc(abmil_roc, abmil_auc,
                        trans_roc, trans_auc,
                        save_path="roc_comparison.png"):
    """Side-by-side ROC curves"""
    if not is_main_process():
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    configs = [
        (axes[0], abmil_roc, abmil_auc, "steelblue", "ABMIL"),
        (axes[1], trans_roc, trans_auc, "seagreen", "TransMIL"),
    ]

    mean_fpr = np.linspace(0, 1, 100)

    for ax, roc_data, mean_auc, color, name in configs:
        if not roc_data:
            continue

        interp_tprs = []
        for i, (fpr, tpr, auc_val) in enumerate(roc_data):
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            ax.plot(fpr, tpr, color=color, alpha=0.25, linewidth=1,
                    label=f"Fold {i + 1} (AUC={auc_val:.3f})")

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(interp_tprs, axis=0)

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
                label=f"Mean ROC (AUC={mean_auc:.3f})")
        ax.fill_between(mean_fpr,
                        np.maximum(mean_tpr - std_tpr, 0),
                        np.minimum(mean_tpr + std_tpr, 1),
                        color=color, alpha=0.15, label="± 1 std")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

        ax.set_title(f"{name}\nMean AUC = {mean_auc:.3f}", fontsize=13)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.set_xlim([0, 1]);
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

    winner = "TransMIL" if trans_auc >= abmil_auc else "ABMIL"
    diff = abs(trans_auc - abmil_auc)
    fig.suptitle(
        f"ABMIL vs TransMIL — {N_FOLDS}-Fold CV Comparison\n"
        f"Winner: {winner}  (ΔAUC = {diff:.3f})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log(f"\nComparison ROC plot saved → {save_path}")


# ─────────────────────────────────────────────
# 7. SAVE / LOAD
# ─────────────────────────────────────────────

def save_model(state_dict, embed_dim, model_name, path=None):
    if not is_main_process():
        return

    if path is None:
        path = f"{model_name}_best.pt"
    torch.save({
        "state_dict": state_dict,
        "embed_dim": embed_dim,
        "model_name": model_name,
        "hidden_dim": HIDDEN_DIM,
        "attn_dim": ATTN_DIM,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
    }, path)
    log(f"  {model_name} best model saved → {path}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    label_dict = load_labels(CSV_PATH, SCORE_THRESH)

    # ── CRITICAL: only rank 0 scans disk and builds the index ──
    # Other ranks receive the index via broadcast
    # NO rank loads actual embedding arrays here — LAZY_LOAD=True handles that
    if is_main_process():
        records = build_dataset(PARQUET_DIR, label_dict)
        EMBED_DIM = int(load_subject_embeddings(records[0][0], max_variants=1).shape[1])
        obj = [records, EMBED_DIM]
    else:
        obj = [None, None]

    # broadcast index (paths + labels) from rank 0 to all ranks
    # this is cheap — paths are just strings, no arrays
    dist.broadcast_object_list(obj, src=0)
    records, EMBED_DIM = obj[0], obj[1]

    # sync all ranks before training begins
    dist.barrier()

    log(f"Embedding dim             : {EMBED_DIM}")
    log(f"Subjects loaded           : {len(records)}")
    log(f"MAX_VARIANTS (ABMIL)      : {MAX_VARIANTS_ABMIL}")
    log(f"MAX_VARIANTS (Transformer): {MAX_VARIANTS_TRANSFORMER}")
    log(f"LAZY_LOAD                 : {LAZY_LOAD} (each rank reads only its own batches)")

    abmil_state, abmil_roc, abmil_auc, abmil_bacc = run_cv("abmil", records, EMBED_DIM)
    save_model(abmil_state, EMBED_DIM, "abmil")

    trans_state, trans_roc, trans_auc, trans_bacc = run_cv("transformer", records, EMBED_DIM)
    save_model(trans_state, EMBED_DIM, "transformer")

    if is_main_process():
        log("\n" + "="*60)
        log("FINAL COMPARISON")
        log(f"  ABMIL     AUC: {abmil_auc:.3f}  BalAcc: {abmil_bacc:.3f}")
        log(f"  TransMIL  AUC: {trans_auc:.3f}  BalAcc: {trans_bacc:.3f}")
        winner = "TransMIL" if trans_auc >= abmil_auc else "ABMIL"
        log(f"  → Winner: {winner}")
        log("="*60)
        plot_comparison_roc(
            abmil_roc, abmil_auc,
            trans_roc, trans_auc,
            save_path="/mnt/data/shyam/aritri/scripts/roc_comparison.png"
        )

    dist.destroy_process_group()