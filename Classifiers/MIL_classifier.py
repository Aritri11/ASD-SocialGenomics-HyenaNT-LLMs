"""
Attention-Based Multiple Instance Learning (ABMIL) Classifier
for Genomic Variant Embeddings → Social Interaction Classification

Problem:
  - Input : ~2000 parquet files, each = one subject (bag of variant embeddings)
  - Target: summary_score from wes_scq_merged.csv
            ≤15 → Typical social interaction  (class 0)
            >15 → Atypical social interaction (class 1)

Architecture choice: Attention-Based MIL (Ilse et al., 2018)
  - Learns per-variant attention weights → interpretable
  - Aggregates variable-length bags into fixed-size representation
  - Works well with N~2000 subjects unlike full Transformers
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  ← edit these paths
# ─────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # GPU 2 — least used, coolest temp

PARQUET_DIR   = "/mnt/data/shyam/aritri/scripts/out_embed"
CSV_PATH      = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"
SCORE_THRESH  = 15
EMBEDDING_COL = "embedding"
MAX_VARIANTS  = 2000
RANDOM_SEED   = 42
N_FOLDS       = 10

# DL hyper-params
EMBED_DIM     = None        # auto-detected from first file
HIDDEN_DIM    = 256
ATTN_DIM      = 128
DROPOUT       = 0.3
EPOCHS        = 50
BATCH_SIZE    = 32
LR            = 1e-4
N_WORKERS     = 8
RF_N_JOBS     = 8

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.4)

LAZY_LOAD = False   # False = load all into RAM (recommended — you have 870GB free)
                    # True  = load from disk on-the-fly (slow, only if RAM is tight)

print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def extract_subject_id(filename: str) -> str:
    m = re.match(r"(SP\d+)", os.path.basename(filename))
    return m.group(1) if m else None


def load_labels(csv_path: str, threshold: int) -> dict:
    """Returns {subject_id: label (0/1)}"""
    df = pd.read_csv(csv_path)
    df["label"] = (df["summary_score"] > threshold).astype(int)
    return dict(zip(df["subject_sp_id"].astype(str), df["label"]))


def load_subject_embeddings(parquet_path: str, max_variants=None) -> np.ndarray:
    """Returns (N_variants, embed_dim) float32 array."""
    df = pd.read_parquet(parquet_path, columns=[EMBEDDING_COL])
    embs = df[EMBEDDING_COL].tolist()
    arr = np.array([np.array(e, dtype=np.float32) for e in embs])
    if max_variants and len(arr) > max_variants:
        idx = np.random.choice(len(arr), max_variants, replace=False)
        arr = arr[idx]
    return arr


def get_embedding_array(record):
    """
    Safely get the embedding array from a record,
    regardless of LAZY_LOAD mode.
    record[0] is either a file path (str) or numpy array.
    """
    if LAZY_LOAD:
        return load_subject_embeddings(record[0], max_variants=MAX_VARIANTS)
    else:
        return record[0]


def build_dataset(parquet_dir: str, label_dict: dict, max_variants=None):
    """
    Scans parquet_dir, matches filenames to label_dict.
    If LAZY_LOAD=False: stores (ndarray, label, sid)
    If LAZY_LOAD=True : stores (path,   label, sid)
    """
    records = []
    files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    print(f"Found {len(files)} parquet files")

    missing = 0
    for fname in sorted(files):
        sid = extract_subject_id(fname)
        if sid is None or sid not in label_dict:
            missing += 1
            continue
        path = os.path.join(parquet_dir, fname)
        try:
            if LAZY_LOAD:
                records.append((path, label_dict[sid], sid))
            else:
                emb = load_subject_embeddings(path, max_variants)
                records.append((emb, label_dict[sid], sid))
        except Exception as ex:
            print(f"  ⚠ Could not load {fname}: {ex}")

    print(f"Loaded {len(records)} subjects  |  skipped {missing} (no label match)")
    labels = [r[1] for r in records]
    print(f"Class distribution → 0 (typical): {labels.count(0)}  "
          f"1 (atypical): {labels.count(1)}")
    return records


# ─────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────

class BagDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # FIX: use get_embedding_array() — works for both LAZY_LOAD modes
        emb = get_embedding_array(self.records[idx])
        label = self.records[idx][1]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def collate_bags(batch):
    bags, labels = zip(*batch)
    return list(bags), torch.stack(labels)


# ─────────────────────────────────────────────
# 3. ABMIL MODEL
# ─────────────────────────────────────────────

class AttentionMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (Ilse et al., 2018)
    https://arxiv.org/abs/1802.04712
    """
    def __init__(self, embed_dim, hidden_dim=256, attn_dim=128,
                 dropout=0.3, n_classes=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
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
        bag_reprs = []
        attn_weights_all = []

        for bag in bags:
            bag = bag.to(next(self.parameters()).device)
            H = self.encoder(bag)

            A_V = torch.tanh(self.attn_V(H))
            A_U = torch.sigmoid(self.attn_U(H))
            A   = self.attn_w(A_V * A_U)
            A   = F.softmax(A, dim=0)

            z = (A * H).sum(dim=0)
            bag_reprs.append(z)
            attn_weights_all.append(A.detach().squeeze())

        bag_reprs = torch.stack(bag_reprs)
        logits = self.classifier(bag_reprs)

        if return_attention:
            return logits, attn_weights_all
        return logits


# ─────────────────────────────────────────────
# 4. TRAINING UTILITIES
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for bags, labels in loader:
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(bags)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for bags, labels in loader:
        logits = model(bags)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────
# 5. BASELINE: RANDOM FOREST ON MEAN EMBEDDING
# ─────────────────────────────────────────────

def run_rf_baseline(records, n_splits=N_FOLDS):
    print("\n" + "="*60)
    print("BASELINE: Random Forest on Mean-Pooled Embeddings")
    print("="*60)

    # FIX: use get_embedding_array() — works for both LAZY_LOAD modes
    X = np.array([get_embedding_array(r).mean(axis=0) for r in records])
    y = np.array([r[1] for r in records])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    aucs, baccs = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            n_jobs=RF_N_JOBS, random_state=RANDOM_SEED  # FIX: use RF_N_JOBS not -1
        )
        clf.fit(Xtr, y[tr])
        probs = clf.predict_proba(Xte)[:, 1]
        preds = clf.predict(Xte)
        auc  = roc_auc_score(y[te], probs)
        bacc = balanced_accuracy_score(y[te], preds)
        aucs.append(auc)
        baccs.append(bacc)
        print(f"  Fold {fold+1}: AUC={auc:.3f}  BalAcc={bacc:.3f}")

    print(f"\n  Mean AUC  : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Mean BalAcc: {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")
    return np.mean(aucs)


# ─────────────────────────────────────────────
# 6. ABMIL CROSS-VALIDATION
# ─────────────────────────────────────────────

def run_abmil_cv(records, embed_dim, n_splits=N_FOLDS):
    print("\n" + "="*60)
    print(f"ATTENTION-MIL Classifier ({n_splits}-fold CV)")
    print("="*60)

    indices = np.arange(len(records))
    labels  = np.array([r[1] for r in records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_aucs, fold_baccs = [], []
    best_model_state = None
    best_auc = 0.0

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n── Fold {fold+1}/{n_splits} ──")
        tr_records = [records[i] for i in tr_idx]
        te_records = [records[i] for i in te_idx]

        n_pos = sum(r[1] for r in tr_records)
        n_neg = len(tr_records) - n_pos
        pos_weight = n_neg / max(n_pos, 1)
        weights = torch.tensor([1.0, pos_weight], device=DEVICE)

        tr_loader = DataLoader(
            BagDataset(tr_records), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_bags,
            num_workers=N_WORKERS, pin_memory=True
        )
        te_loader = DataLoader(
            BagDataset(te_records), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_bags,
            num_workers=N_WORKERS, pin_memory=True
        )

        model = AttentionMIL(
            embed_dim=embed_dim,
            hidden_dim=HIDDEN_DIM,
            attn_dim=ATTN_DIM,
            dropout=DROPOUT
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_val_auc, patience_cnt = 0.0, 0
        PATIENCE = 10
        fold_state = None

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion)
            scheduler.step()
            y_true, y_pred, y_prob = eval_epoch(model, te_loader)
            val_auc  = roc_auc_score(y_true, y_prob)
            val_bacc = balanced_accuracy_score(y_true, y_pred)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d} | loss={tr_loss:.4f} acc={tr_acc:.3f} "
                      f"| val AUC={val_auc:.3f} BalAcc={val_bacc:.3f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_cnt = 0
                fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

        # reload best weights for final eval
        if fold_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in fold_state.items()})

        y_true, y_pred, y_prob = eval_epoch(model, te_loader)
        auc  = roc_auc_score(y_true, y_prob)
        bacc = balanced_accuracy_score(y_true, y_pred)
        fold_aucs.append(auc)
        fold_baccs.append(bacc)

        print(f"\n  Fold {fold+1} final → AUC={auc:.3f}  BalAcc={bacc:.3f}")
        print(classification_report(y_true, y_pred,
              target_names=["Typical (≤15)", "Atypical (>15)"]))

        if auc > best_auc:
            best_auc = auc
            best_model_state = fold_state

    print("\n" + "="*60)
    print(f"ABMIL CV SUMMARY")
    print(f"  Mean AUC   : {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"  Mean BalAcc: {np.mean(fold_baccs):.3f} ± {np.std(fold_baccs):.3f}")
    print("="*60)
    return best_model_state, embed_dim


# ─────────────────────────────────────────────
# 7. SAVE + INFERENCE HELPER
# ─────────────────────────────────────────────

def save_model(state_dict, embed_dim, path="abmil_best.pt"):
    torch.save({"state_dict": state_dict, "embed_dim": embed_dim,
                "hidden_dim": HIDDEN_DIM, "attn_dim": ATTN_DIM}, path)
    print(f"\nBest model saved → {path}")


def load_model_for_inference(path="abmil_best.pt"):
    ckpt = torch.load(path, map_location=DEVICE)
    model = AttentionMIL(
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        attn_dim=ckpt["attn_dim"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_subject(model, parquet_path: str):
    """
    Returns (predicted_class, probability_atypical, attention_weights_array)
    """
    emb = load_subject_embeddings(parquet_path, max_variants=MAX_VARIANTS)
    bag = torch.tensor(emb, dtype=torch.float32)
    logits, attn = model([bag], return_attention=True)
    prob_atypical = F.softmax(logits, dim=1)[0, 1].item()
    pred_class    = int(logits.argmax(1).item())
    attn_np       = attn[0].cpu().numpy()
    return pred_class, prob_atypical, attn_np


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ── load labels ──
    label_dict = load_labels(CSV_PATH, SCORE_THRESH)
    print(f"Labels loaded: {len(label_dict)} subjects")

    # ── build dataset ──
    records = build_dataset(PARQUET_DIR, label_dict, max_variants=MAX_VARIANTS)
    if len(records) == 0:
        raise RuntimeError("No matching records found. Check PARQUET_DIR / CSV_PATH / subject ID regex.")

    # ── auto-detect embedding dim — FIX: works for both LAZY_LOAD modes ──
    if LAZY_LOAD:
        sample_emb = load_subject_embeddings(records[0][0], max_variants=1)
        EMBED_DIM = sample_emb.shape[1]
    else:
        EMBED_DIM = records[0][0].shape[1]
    print(f"Embedding dim: {EMBED_DIM}")

    # ── baseline RF ──
    rf_auc = run_rf_baseline(records)

    # ── ABMIL ──
    best_state, embed_dim = run_abmil_cv(records, EMBED_DIM)

    # ── save best model ──
    save_model(best_state, embed_dim)

    # ── example inference ──
    print("\n── Example inference ──")
    example_file = os.path.join(PARQUET_DIR, os.listdir(PARQUET_DIR)[0])
    model = load_model_for_inference("abmil_best.pt")
    pred, prob, attn = predict_subject(model, example_file)
    print(f"File        : {os.path.basename(example_file)}")
    print(f"Prediction  : {'Atypical (>15)' if pred == 1 else 'Typical (≤15)'}")
    print(f"P(atypical) : {prob:.3f}")
    print(f"Top-5 most attended variant indices: {attn.argsort()[-5:][::-1]}")


#Results:
# Using device: cuda
# Labels loaded: 40537 subjects
# Found 1779 parquet files
# Loaded 1779 subjects  |  skipped 0 (no label match)
# Class distribution → 0 (typical): 773  1 (atypical): 1006
# Embedding dim: 1280
#
# ============================================================
# BASELINE: Random Forest on Mean-Pooled Embeddings
# ============================================================
#   Fold 1: AUC=0.660  BalAcc=0.631
#   Fold 2: AUC=0.690  BalAcc=0.618
#   Fold 3: AUC=0.566  BalAcc=0.515
#   Fold 4: AUC=0.527  BalAcc=0.521
#   Fold 5: AUC=0.632  BalAcc=0.600
#   Fold 6: AUC=0.625  BalAcc=0.566
#   Fold 7: AUC=0.533  BalAcc=0.542
#   Fold 8: AUC=0.584  BalAcc=0.564
#   Fold 9: AUC=0.484  BalAcc=0.498
#   Fold 10: AUC=0.540  BalAcc=0.568
#
#   Mean AUC  : 0.584 ± 0.063
#   Mean BalAcc: 0.562 ± 0.042
#
# ============================================================
# ATTENTION-MIL Classifier (10-fold CV)
# ============================================================
#
# ── Fold 1/10 ──
#   Ep   1 | loss=0.7009 acc=0.498 | val AUC=0.544 BalAcc=0.500
#   Ep  10 | loss=0.6935 acc=0.488 | val AUC=0.509 BalAcc=0.500
#   Early stop at epoch 17
#
#   Fold 1 final → AUC=0.560  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.43      1.00      0.60        77
# Atypical (>15)       0.00      0.00      0.00       101
#
#       accuracy                           0.43       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.43      0.26       178
#
#
# ── Fold 2/10 ──
#   Ep   1 | loss=0.7007 acc=0.499 | val AUC=0.476 BalAcc=0.500
#   Ep  10 | loss=0.6962 acc=0.520 | val AUC=0.492 BalAcc=0.500
#   Early stop at epoch 15
#
#   Fold 2 final → AUC=0.506  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.43      1.00      0.60        77
# Atypical (>15)       0.00      0.00      0.00       101
#
#       accuracy                           0.43       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.43      0.26       178
#
#
# ── Fold 3/10 ──
#   Ep   1 | loss=0.6972 acc=0.472 | val AUC=0.539 BalAcc=0.500
#   Ep  10 | loss=0.6940 acc=0.521 | val AUC=0.554 BalAcc=0.500
#   Early stop at epoch 16
#
#   Fold 3 final → AUC=0.620  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.43      1.00      0.60        77
# Atypical (>15)       0.00      0.00      0.00       101
#
#       accuracy                           0.43       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.43      0.26       178
#
#
# ── Fold 4/10 ──
#   Ep   1 | loss=0.6975 acc=0.492 | val AUC=0.544 BalAcc=0.500
#   Ep  10 | loss=0.6935 acc=0.532 | val AUC=0.457 BalAcc=0.500
#   Ep  20 | loss=0.6942 acc=0.481 | val AUC=0.526 BalAcc=0.500
#   Early stop at epoch 23
#
#   Fold 4 final → AUC=0.546  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.00      0.00      0.00        77
# Atypical (>15)       0.57      1.00      0.72       101
#
#       accuracy                           0.57       178
#      macro avg       0.28      0.50      0.36       178
#   weighted avg       0.32      0.57      0.41       178
#
#
# ── Fold 5/10 ──
#   Ep   1 | loss=0.6960 acc=0.507 | val AUC=0.553 BalAcc=0.500
#   Ep  10 | loss=0.6943 acc=0.503 | val AUC=0.551 BalAcc=0.500
#   Ep  20 | loss=0.6937 acc=0.519 | val AUC=0.587 BalAcc=0.500
#   Ep  30 | loss=0.6943 acc=0.514 | val AUC=0.567 BalAcc=0.500
#   Early stop at epoch 30
#
#   Fold 5 final → AUC=0.587  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.00      0.00      0.00        77
# Atypical (>15)       0.57      1.00      0.72       101
#
#       accuracy                           0.57       178
#      macro avg       0.28      0.50      0.36       178
#   weighted avg       0.32      0.57      0.41       178
#
#
# ── Fold 6/10 ──
#   Ep   1 | loss=0.6975 acc=0.510 | val AUC=0.440 BalAcc=0.500
#   Ep  10 | loss=0.6955 acc=0.504 | val AUC=0.388 BalAcc=0.500
#   Early stop at epoch 16
#
#   Fold 6 final → AUC=0.581  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.43      1.00      0.60        77
# Atypical (>15)       0.00      0.00      0.00       101
#
#       accuracy                           0.43       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.43      0.26       178
#
#
# ── Fold 7/10 ──
#   Ep   1 | loss=0.6992 acc=0.496 | val AUC=0.486 BalAcc=0.500
#   Ep  10 | loss=0.6956 acc=0.503 | val AUC=0.461 BalAcc=0.500
#   Early stop at epoch 12
#
#   Fold 7 final → AUC=0.536  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.44      1.00      0.61        78
# Atypical (>15)       0.00      0.00      0.00       100
#
#       accuracy                           0.44       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.44      0.27       178
#
#
# ── Fold 8/10 ──
#   Ep   1 | loss=0.6985 acc=0.495 | val AUC=0.527 BalAcc=0.500
#   Ep  10 | loss=0.6969 acc=0.505 | val AUC=0.451 BalAcc=0.500
#   Early stop at epoch 11
#
#   Fold 8 final → AUC=0.527  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.44      1.00      0.61        78
# Atypical (>15)       0.00      0.00      0.00       100
#
#       accuracy                           0.44       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.44      0.27       178
#
#
# ── Fold 9/10 ──
#   Ep   1 | loss=0.6974 acc=0.498 | val AUC=0.504 BalAcc=0.500
#   Ep  10 | loss=0.6943 acc=0.489 | val AUC=0.505 BalAcc=0.500
#   Early stop at epoch 16
#
#   Fold 9 final → AUC=0.544  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.44      1.00      0.61        78
# Atypical (>15)       0.00      0.00      0.00       100
#
#       accuracy                           0.44       178
#      macro avg       0.22      0.50      0.30       178
#   weighted avg       0.19      0.44      0.27       178
#
#
# ── Fold 10/10 ──
#   Ep   1 | loss=0.6965 acc=0.507 | val AUC=0.495 BalAcc=0.500
#   Ep  10 | loss=0.6946 acc=0.493 | val AUC=0.502 BalAcc=0.500
#   Early stop at epoch 12
#
#   Fold 10 final → AUC=0.560  BalAcc=0.500
#                 precision    recall  f1-score   support
#
#  Typical (≤15)       0.44      1.00      0.61        77
# Atypical (>15)       0.00      0.00      0.00       100
#
#       accuracy                           0.44       177
#      macro avg       0.22      0.50      0.30       177
#   weighted avg       0.19      0.44      0.26       177
#
#
# ============================================================
# ABMIL CV SUMMARY
#   Mean AUC   : 0.557 ± 0.031
#   Mean BalAcc: 0.500 ± 0.000
# ============================================================
#
# Best model saved → abmil_best.pt
#
# ── Example inference ──
# File        : SP0018133.gvcf_variants.parquet
# Prediction  : Typical (≤15)
# P(atypical) : 0.495
# Top-5 most attended variant indices: [1847  461  523 1187   62]
