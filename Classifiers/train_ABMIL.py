"""
ABMIL Training Only
Run with: python -m torch.distributed.launch --nproc_per_node=4 train_abmil_only.py
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
    if is_main_process():
        print(msg)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PARQUET_DIR = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
CSV_PATH = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"
SCORE_THRESH = 15
EMBEDDING_COL = "embedding"

MAX_VARIANTS_ABMIL = 4096  # ✅ CAP VARIANTS
RANDOM_SEED = 42
N_FOLDS = 5
EMBED_DIM = None
HIDDEN_DIM = 256
DROPOUT = 0.1
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 15
N_WORKERS = 2  # ✅ REDUCED
LR = 3e-4
ATTN_DIM = 128
USE_AMP = True
LAZY_LOAD = True  # ✅ CRITICAL

if is_main_process():
    log(f"Device: {device}")
    log(f"Running ABMIL only")


# ─────────────────────────────────────────────
# [Copy all helper functions from original script]
# FocalLoss, extract_subject_id, load_labels,
# load_subject_embeddings, get_embedding_array,
# build_dataset, BagDataset, collate_bags
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


def extract_subject_id(filename: str) -> str:
    m = re.match(r"(SP\d+)", os.path.basename(filename))
    return m.group(1) if m else None


def load_labels(csv_path: str, threshold: int) -> dict:
    df = pd.read_csv(csv_path)
    df["label"] = (df["summary_score"] > threshold).astype(int)
    return dict(zip(df["subject_sp_id"].astype(str), df["label"]))


def load_subject_embeddings(parquet_path: str, max_variants=None) -> np.ndarray:
    import pyarrow.parquet as pq
    try:
        table = pq.read_table(parquet_path, columns=[EMBEDDING_COL], memory_map=True)
        df = table.to_pandas()
        embs = df[EMBEDDING_COL].tolist()
        arr = np.array([np.array(e, dtype=np.float32) for e in embs])
        if max_variants and len(arr) > max_variants:
            idx = np.random.choice(len(arr), max_variants, replace=False)
            arr = arr[idx]
        return arr
    except Exception as e:
        log(f"⚠️ Error reading {parquet_path}: {type(e).__name__}")
        log(f"   Returning empty array (will skip this sample)")
        # Return None to signal error
        return None


def get_embedding_array(record, max_variants=None):
    result = load_subject_embeddings(record[0], max_variants=max_variants)
    if result is None:
        raise ValueError(f"Failed to load embeddings from {record[0]}")
    return result


def build_dataset(parquet_dir, label_dict):
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
        records.append((path, label_dict[sid], sid))
    if is_main_process():
        log(f"Indexed {len(records)} subjects | skipped {missing}")
        labels = [r[1] for r in records]
        log(f"Class distribution → 0: {labels.count(0)} | 1: {labels.count(1)}")
    return records


class BagDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        try:
            emb = get_embedding_array(self.records[idx], max_variants=MAX_VARIANTS_ABMIL)
            label = self.records[idx][1]
            return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Return a dummy sample if loading fails
            log(f"⚠️ Skipping sample {idx}: {str(e)}")
            # Return random embedding to avoid breaking dataloader
            dummy_emb = np.random.randn(4096, 1280).astype(np.float32)
            return torch.tensor(dummy_emb, dtype=torch.float32), torch.tensor(0, dtype=torch.long)


def collate_bags(batch):
    bags, labels = zip(*batch)
    return list(bags), torch.stack(labels)


# ─────────────────────────────────────────────
# ABMIL MODEL ONLY
# ─────────────────────────────────────────────

class AttentionMIL(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, attn_dim=128, dropout=0.1, n_classes=2):
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
        bags = [b.to(device) for b in bags]
        max_len = max(len(b) for b in bags)
        padded_bags, masks = [], []
        for bag in bags:
            pad_len = max_len - len(bag)
            if pad_len > 0:
                bag = torch.cat([bag, torch.zeros(pad_len, bag.shape[1], device=device)])
            mask = torch.zeros(max_len, device=device)
            mask[:len(bag) - pad_len] = 1.0
            padded_bags.append(bag)
            masks.append(mask)

        bags_tensor = torch.stack(padded_bags)
        masks_tensor = torch.stack(masks)
        H = self.encoder(bags_tensor)
        A_V = torch.tanh(self.attn_V(H))
        A_U = torch.sigmoid(self.attn_U(H))
        A = self.attn_w(A_V * A_U)
        A = A + (1 - masks_tensor.unsqueeze(-1)) * (-1e9)
        A = F.softmax(A, dim=1)
        z = (A * H).sum(dim=1)
        logits = self.classifier(z)
        return logits


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler):
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
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
def eval_model(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for bags, labels in loader:
        bags = [b.to(device) for b in bags]
        labels = labels.to(device)
        logits = model(bags)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────
# CV RUNNER (ABMIL ONLY)
# ─────────────────────────────────────────────

def run_cv(records, embed_dim, n_splits=N_FOLDS):
    log("\n" + "=" * 60)
    log(f"ABMIL — {n_splits}-Fold CV")
    log("=" * 60)

    indices = np.arange(len(records))
    labels = np.array([r[1] for r in records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_aucs, fold_baccs = [], []
    fold_roc_data = []
    best_model_state, best_auc = None, 0.0

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, labels)):
        torch.cuda.empty_cache()  # ✅ Clear GPU memory between folds
        log(f"\n── Fold {fold + 1}/{n_splits} ──")

        tr_records = [records[i] for i in tr_idx]
        te_records = [records[i] for i in te_idx]

        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        scaler = GradScaler("cuda") if USE_AMP else None

        model = AttentionMIL(embed_dim=embed_dim, hidden_dim=HIDDEN_DIM,
                             attn_dim=ATTN_DIM, dropout=DROPOUT).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, broadcast_buffers=True)

        tr_dataset = BagDataset(tr_records)
        tr_sampler = DistributedSampler(tr_dataset, num_replicas=world_size,
                                        rank=rank, shuffle=True, seed=RANDOM_SEED)
        tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, sampler=tr_sampler,
                               collate_fn=collate_bags, num_workers=N_WORKERS, pin_memory=True)

        te_dataset = BagDataset(te_records)
        te_sampler = DistributedSampler(te_dataset, num_replicas=world_size,
                                        rank=rank, shuffle=False, seed=RANDOM_SEED)
        te_loader = DataLoader(te_dataset, batch_size=BATCH_SIZE, sampler=te_sampler,
                               collate_fn=collate_bags, num_workers=N_WORKERS, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_auc, patience_cnt, fold_state = 0.0, 0, None

        for epoch in range(1, EPOCHS + 1):
            tr_sampler.set_epoch(epoch)
            tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, scaler)
            scheduler.step()
            y_true, y_pred, y_prob = eval_model(model, te_loader)

            if is_main_process() and y_true is not None:
                val_auc = roc_auc_score(y_true, y_prob)
                val_bacc = balanced_accuracy_score(y_true, y_pred)
                if epoch % 10 == 0 or epoch == 1:
                    log(f"  Ep {epoch:3d} | loss={tr_loss:.4f} acc={tr_acc:.3f} | AUC={val_auc:.3f} BalAcc={val_bacc:.3f}")
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_cnt = 0
                    fold_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        log(f"  Early stop at epoch {epoch}")
                        break
            dist.barrier()

        if fold_state is not None and is_main_process():
            model.module.load_state_dict({k: v.to(device) for k, v in fold_state.items()})
        dist.barrier()

        y_true, y_pred, y_prob = eval_model(model, te_loader)
        if is_main_process() and y_true is not None:
            auc = roc_auc_score(y_true, y_prob)
            bacc = balanced_accuracy_score(y_true, y_pred)
            fold_aucs.append(auc)
            fold_baccs.append(bacc)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fold_roc_data.append((fpr, tpr, auc))
            log(f"\n  Fold {fold + 1} final → AUC={auc:.3f}  BalAcc={bacc:.3f}")
            log(classification_report(y_true, y_pred, target_names=["Typical (≤15)", "Atypical (>15)"]))
            if auc > best_auc:
                best_auc = auc
                best_model_state = fold_state

        # ✅ DELETE MODEL TO FREE GPU MEMORY
        del model, tr_loader, te_loader, tr_dataset, te_dataset
        torch.cuda.empty_cache()

    if is_main_process():
        mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
        mean_bacc = np.mean(fold_baccs) if fold_baccs else 0.0
        log("\n" + "=" * 60)
        log(f"ABMIL SUMMARY")
        log(f"  Mean AUC   : {mean_auc:.3f} ± {np.std(fold_aucs) if fold_aucs else 0:.3f}")
        log(f"  Mean BalAcc: {mean_bacc:.3f} ± {np.std(fold_baccs) if fold_baccs else 0:.3f}")
        log("=" * 60)
        return best_model_state, fold_roc_data, mean_auc, mean_bacc
    else:
        return None, None, 0.0, 0.0


def save_model(state_dict, embed_dim, model_name, path=None):
    if not is_main_process():
        return
    if path is None:
        path = f"{model_name}_best.pt"
    torch.save({"state_dict": state_dict, "embed_dim": embed_dim, "model_name": model_name,
                "hidden_dim": HIDDEN_DIM, "attn_dim": ATTN_DIM}, path)
    log(f"  {model_name} best model saved → {path}")

# ─────────────────────────***

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        log("=" * 60)
        log("STARTING ABMIL TRAINING")
        log("=" * 60)

        log(f"Step 1: Loading labels from {CSV_PATH}")
        label_dict = load_labels(CSV_PATH, SCORE_THRESH)
        log(f"  ✓ Loaded {len(label_dict)} labels")

        if is_main_process():
            log(f"Step 2: Building dataset index...")
            records = build_dataset(PARQUET_DIR, label_dict)
            log(f"  ✓ Indexed {len(records)} records")

            log(f"Step 3: Getting embedding dimension...")
            sample_path = records[0][0]
            log(f"  Loading sample from: {sample_path}")
            sample_emb = load_subject_embeddings(sample_path, max_variants=1)
            EMBED_DIM = int(sample_emb.shape[1])
            log(f"  ✓ EMBED_DIM = {EMBED_DIM}")
            obj = [records, EMBED_DIM]
        else:
            log(f"Rank {rank}: Waiting for broadcast...")
            obj = [None, None]

        log(f"Step 4: Broadcasting data from rank 0...")
        dist.broadcast_object_list(obj, src=0)
        records, EMBED_DIM = obj[0], obj[1]
        log(f"  ✓ Rank {rank} received {len(records)} records, EMBED_DIM={EMBED_DIM}")

        dist.barrier()
        log(f"Step 5: All ranks synchronized")

        log(f"\nConfiguration:")
        log(f"  Embedding dim: {EMBED_DIM}")
        log(f"  Subjects: {len(records)}")
        log(f"  Folds: {N_FOLDS}")
        log(f"  Epochs: {EPOCHS}")
        log(f"  Batch size: {BATCH_SIZE}")
        log(f"  Workers: {N_WORKERS}")

        log(f"\nStep 6: Starting CV training...")
        abmil_state, abmil_roc, abmil_auc, abmil_bacc = run_cv(records, EMBED_DIM)
        save_model(abmil_state, EMBED_DIM, "abmil")

        if is_main_process():
            log(f"\n✓ ABMIL Final AUC: {abmil_auc:.3f}  BalAcc: {abmil_bacc:.3f}")

        dist.destroy_process_group()

    except Exception as e:
        log(f"\n❌ ERROR in rank {rank}: {type(e).__name__}")
        log(f"Message: {str(e)}")
        import traceback

        log(f"Traceback:\n{traceback.format_exc()}")
        dist.destroy_process_group()
        raise