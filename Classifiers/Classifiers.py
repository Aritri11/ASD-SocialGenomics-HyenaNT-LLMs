"""
Classical ML Classifiers for Social Communication Classification
Random Forest + SVM + XGBoost on genomic variant embeddings

Input:
  - Parquet files: variant embeddings per subject
  - CSV file    : SCQ scores per subject

Classification:
  SCQ score <= 15 → Typical social communication    (class 0)
  SCQ score >  15 → Atypical social communication   (class 1)

Pooling strategy: ATTENTION POOLING
  Instead of mean pooling (treats all variants equally),
  attention pooling learns a scalar weight per variant
  so diagnostically relevant variants contribute more
  to the final subject representation.

  Since RF/SVM/XGBoost require fixed-size inputs, we train
  a lightweight attention network first to pool each subject's
  ~46k variant embeddings → single (embed_dim,) vector.
  This pooled vector is then used as input to the classifiers.

Output per model:
  - Classification report (precision, recall, F1)
  - AUC-ROC score
  - ROC curve plot (saved as PNG)

Run:
  python ml_classifiers_attn.py
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Run: pip install xgboost")

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PARQUET_DIR   = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
CSV_PATH      = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"
OUT_DIR       = "/mnt/data/shyam/aritri/scripts/ml_results"
SCORE_THRESH  = 15           # <= 15 → typical (0), > 15 → atypical (1)
EMBEDDING_COL = "embedding"
MAX_VARIANTS  = None         # None = use all variants
RANDOM_SEED   = 42
N_FOLDS       = 5            # stratified K-fold CV
N_JOBS        = 8            # CPU cores for RF and XGBoost

# ── Attention pooling training config ──
ATTN_HIDDEN   = 128          # hidden dim of attention network
ATTN_EPOCHS   = 20           # epochs to train attention pooler
ATTN_LR       = 3e-4         # learning rate
ATTN_BATCH    = 32           # subjects per batch
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


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
    label_dict  = dict(zip(df["subject_sp_id"].astype(str), df["label"]))
    print(f"Labels loaded: {len(label_dict)} subjects from CSV")
    n_pos = sum(label_dict.values())
    n_neg = len(label_dict) - n_pos
    print(f"  Typical   (<=15, class 0): {n_neg}")
    print(f"  Atypical  (> 15, class 1): {n_pos}")
    return label_dict


def is_valid_parquet(path: str) -> bool:
    """
    Quick check — reads only the footer metadata (fast, ~1ms per file).
    Catches corrupted files before they crash training 12 hours later.
    """
    try:
        meta = pq.read_metadata(path)
        return meta.num_rows > 0
    except Exception:
        return False


def load_raw_embeddings(parquet_path: str,
                         max_variants=None) -> np.ndarray:
    """Returns (N_variants, embed_dim) float32 array."""
    try:
        df  = pd.read_parquet(parquet_path, columns=[EMBEDDING_COL])
        emb = np.array(df[EMBEDDING_COL].tolist(), dtype=np.float32)
        if max_variants and len(emb) > max_variants:
            idx = np.random.choice(len(emb), max_variants, replace=False)
            emb = emb[idx]
        return emb
    except Exception as e:
        raise RuntimeError(f"Failed to read {parquet_path}: {e}")


def compute_mean_pooled(parquet_path: str,
                         max_variants=None) -> np.ndarray:
    """
    Streams one parquet file, computes mean embedding, discards raw data.
    RAM usage: only (embed_dim,) = 1280 floats per subject instead of
               (46000, 1280) = 59M floats.
    """
    emb = load_raw_embeddings(parquet_path, max_variants)
    return emb.mean(axis=0)   # (embed_dim,) — tiny


# ─────────────────────────────────────────────
# 2. ATTENTION POOLING NETWORK
# ─────────────────────────────────────────────

class AttentionPooler(nn.Module):
    """
    Learns a scalar attention weight per variant then
    produces a weighted sum as the subject representation.

    Why attention pooling over mean pooling?
      Mean pooling: all variants contribute equally
      Attention pooling: learns which variants matter more
                    → rare pathogenic variants can dominate
                    → noise variants get near-zero weight
                    → biologically more meaningful

    Architecture:
      variant_i (D,) → Linear(D→H) → Tanh → Linear(H→1) → scalar weight
      weights = Softmax over all N variants
      subject_repr = Σ weight_i × variant_i    (D,)
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D) — all variant embeddings for one subject
        returns: (D,) — attention-weighted subject representation
        """
        scores  = self.attention(x)              # (N, 1)
        weights = torch.softmax(scores, dim=0)   # (N, 1)
        pooled  = (weights * x).sum(dim=0)       # (D,)
        return pooled


class StreamingSubjectDataset(Dataset):
    """
    Loads parquet files on-demand during attention pooler training.
    Never stores raw embeddings in RAM — only paths and labels.
    RAM usage: O(batch_size) instead of O(all_subjects).
    """
    def __init__(self, path_label_pairs, max_variants=None):
        self.pairs        = path_label_pairs   # list of (path, label)
        self.max_variants = max_variants

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        try:
            emb = load_raw_embeddings(path, self.max_variants)
            return (
                torch.tensor(emb, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            )
        except Exception as e:
            # return None for corrupted files — filtered in collate
            warnings.warn(f"[SKIP] Could not read {os.path.basename(path)}: {e}")
            return None


def collate_variable_length(batch):
    """Filters None entries (corrupted files) and returns valid batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None   # entire batch was corrupted — handled in training loop
    embs, labels = zip(*batch)
    return list(embs), torch.stack(labels)


def train_attention_pooler(path_label_pairs_train,
                            embed_dim: int,
                            epochs: int = ATTN_EPOCHS) -> AttentionPooler:
    """
    Trains attention pooler on training subjects only.
    Streams data from disk — no RAM accumulation.
    """
    print(f"  Training attention pooler on {len(path_label_pairs_train)} "
          f"subjects ({epochs} epochs)...")

    pooler     = AttentionPooler(embed_dim, ATTN_HIDDEN).to(DEVICE)
    classifier = nn.Linear(embed_dim, 2).to(DEVICE)
    optimizer  = torch.optim.Adam(
        list(pooler.parameters()) + list(classifier.parameters()),
        lr=ATTN_LR
    )
    criterion  = nn.CrossEntropyLoss()

    dataset = StreamingSubjectDataset(
        path_label_pairs_train, max_variants=MAX_VARIANTS
    )
    loader  = DataLoader(
        dataset, batch_size=ATTN_BATCH,
        shuffle=True, collate_fn=collate_variable_length,
        num_workers=0,   # ← 0 = main process reads files
                         # avoids BlockingIOError on shared servers
                         # with many concurrent file descriptors
        pin_memory=False
    )

    pooler.train()
    classifier.train()

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            if batch is None:         # ← entire batch was corrupted
                continue
            embs, labels = batch
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            pooled = torch.stack([
                pooler(e.to(DEVICE)) for e in embs
            ])                                       # (B, D)
            logits = classifier(pooled)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | "
                  f"loss={total_loss/max(total,1):.4f} | "
                  f"acc={correct/max(total,1):.3f}")

    pooler.eval()
    return pooler


@torch.no_grad()
def extract_attn_features(path_label_pairs,
                            pooler: AttentionPooler) -> np.ndarray:
    """
    Streams parquet files one by one, applies attention pooler,
    returns (n_subjects, embed_dim) feature matrix.
    RAM at any point: only one subject's embeddings in memory.
    """
    pooler.eval()
    features = []
    for path, _ in tqdm(path_label_pairs, desc="  Extracting features"):
        emb = load_raw_embeddings(path, MAX_VARIANTS)
        t   = torch.tensor(emb, dtype=torch.float32).to(DEVICE)
        pooled = pooler(t).cpu().numpy()
        features.append(pooled)
        del emb, t   # explicitly free memory
    return np.array(features, dtype=np.float32)


def build_index(parquet_dir: str, label_dict: dict):
    """
    Builds a lightweight index of (path, label) pairs.
    Validates each file's metadata during indexing so corrupted
    files are caught NOW — not 12 hours into training.
    RAM usage: negligible (strings only).
    """
    files   = sorted([f for f in os.listdir(parquet_dir)
                      if f.endswith(".parquet")])
    print(f"\nFound {len(files)} parquet files")
    print(f"Validating files during indexing (metadata check)...")

    pairs    = []
    skipped  = 0
    corrupt  = 0

    for fname in tqdm(files, desc="Indexing + validating"):
        sid = extract_subject_id(fname)
        if sid is None or sid not in label_dict:
            skipped += 1
            continue
        path = os.path.join(parquet_dir, fname)

        # ← validate NOW so corrupted files never reach training
        if not is_valid_parquet(path):
            corrupt += 1
            continue

        pairs.append((path, label_dict[sid]))

    labels = np.array([p[1] for p in pairs])
    print(f"Indexed {len(pairs)} valid subjects")
    print(f"  Skipped (no label match) : {skipped}")
    print(f"  Skipped (corrupted)      : {corrupt}")
    print(f"Class 0 (typical)          : {(labels == 0).sum()}")
    print(f"Class 1 (atypical)         : {(labels == 1).sum()}")
    print(f"RAM used                   : ~{len(pairs) * 200 / 1e6:.1f} MB "
          f"(paths only — no embeddings loaded)")

    # detect embed_dim from one file only
    sample    = load_raw_embeddings(pairs[0][0], max_variants=1)
    embed_dim = sample.shape[1]
    print(f"Embed dim                  : {embed_dim}")

    return pairs, labels, embed_dim


# ─────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────

def get_models():
    """
    Returns dict of {name: sklearn pipeline}.
    Each pipeline: StandardScaler → Classifier
    StandardScaler is important for SVM.
    RF and XGBoost are scale-invariant but it doesn't hurt.
    """
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators    = 500,
                max_depth       = None,
                class_weight    = "balanced",   # handles class imbalance
                n_jobs          = N_JOBS,
                random_state    = RANDOM_SEED,
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel       = "rbf",
                C            = 1.0,
                gamma        = "scale",
                class_weight = "balanced",      # handles class imbalance
                probability  = True,            # needed for AUC-ROC
                random_state = RANDOM_SEED,
            ))
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators       = 500,
                max_depth          = 6,
                learning_rate      = 0.05,
                subsample          = 0.8,
                colsample_bytree   = 0.8,
                use_label_encoder  = False,
                eval_metric        = "logloss",
                scale_pos_weight   = 1,         # adjusted below per fold
                n_jobs             = N_JOBS,
                random_state       = RANDOM_SEED,
            ))
        ])
    else:
        print("⚠ XGBoost not available — skipping. Install with: pip install xgboost")

    return models


# ─────────────────────────────────────────────
# 4. EVALUATION (CV with per-fold attention pooling)
# ─────────────────────────────────────────────

def run_cv_evaluation(name, pipeline, pairs, y, embed_dim,
                       n_splits=N_FOLDS):
    """
    Stratified K-fold CV with streaming attention pooling.

    Per fold:
      1. Split (path, label) index into train/test — no data loaded yet
      2. Train AttentionPooler streaming from disk (train fold only)
      3. Stream-extract attention features for train + test
      4. Fit RF/SVM/XGBoost on train features
      5. Evaluate on test features
      6. Free all large arrays before next fold

    RAM at any point: O(batch_size × variants) not O(all_subjects × variants)
    """
    print(f"\n{'='*60}")
    print(f"Model: {name}  (streaming attention pooling)")
    print(f"{'='*60}")

    indices    = np.arange(len(pairs))
    skf        = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RANDOM_SEED)
    fold_aucs, fold_baccs = [], []
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, y)):
        print(f"\n  ── Fold {fold+1}/{n_splits} ──")

        tr_pairs = [pairs[i] for i in tr_idx]
        te_pairs = [pairs[i] for i in te_idx]
        y_tr     = y[tr_idx]
        y_te     = y[te_idx]

        # ── step 1: train attention pooler (streams from disk) ──
        pooler = train_attention_pooler(tr_pairs, embed_dim)

        # ── step 2: extract features (streams from disk) ──
        X_tr = extract_attn_features(tr_pairs, pooler)
        X_te = extract_attn_features(te_pairs, pooler)

        # ── step 3: per-fold XGBoost imbalance adjustment ──
        if name == "XGBoost" and XGBOOST_AVAILABLE:
            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            pipeline.named_steps["clf"].set_params(
                scale_pos_weight=n_neg / max(n_pos, 1)
            )

        # ── step 4: fit and evaluate ──
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        auc  = roc_auc_score(y_te, y_prob)
        bacc = balanced_accuracy_score(y_te, y_pred)
        fold_aucs.append(auc)
        fold_baccs.append(bacc)
        all_y_true.extend(y_te)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1}: AUC={auc:.3f}  BalAcc={bacc:.3f}")

        # ── step 5: free large arrays explicitly ──
        del X_tr, X_te, pooler

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)
    mean_auc   = np.mean(fold_aucs)
    std_auc    = np.std(fold_aucs)
    mean_bacc  = np.mean(fold_baccs)
    std_bacc   = np.std(fold_baccs)

    print(f"\n  Mean AUC    : {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  Mean BalAcc : {mean_bacc:.3f} ± {std_bacc:.3f}")
    print(f"\nClassification Report (aggregated over all folds):")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=["Typical (<=15)", "Atypical (>15)"]
    ))
    cm = confusion_matrix(all_y_true, all_y_pred)
    TP = cm[1,1]; TN = cm[0,0]; FP = cm[0,1]; FN = cm[1,0]
    print(f"  TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")

    return {
        "name"      : name,
        "y_true"    : all_y_true,
        "y_prob"    : all_y_prob,
        "y_pred"    : all_y_pred,
        "mean_auc"  : mean_auc,
        "std_auc"   : std_auc,
        "mean_bacc" : mean_bacc,
        "std_bacc"  : std_bacc,
        "fold_aucs" : fold_aucs,
    }


# ─────────────────────────────────────────────
# 5. PLOTTING
# ─────────────────────────────────────────────

def plot_roc_curves(results, save_path):
    """
    Plots ROC curves for all models on a single figure.
    """
    colors = {
        "Random Forest" : "steelblue",
        "SVM"           : "darkorange",
        "XGBoost"       : "seagreen",
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for res in results:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        color = colors.get(res["name"], "gray")
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{res['name']}  "
                      f"(AUC={res['mean_auc']:.3f} ± {res['std_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(
        "ROC Curves — ML Classifiers\n"
        "Typical vs Atypical Social Communication (SCQ threshold=15)",
        fontsize=12
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nROC curve saved → {save_path}")


def plot_confusion_matrices(results, save_path):
    """
    Plots confusion matrices for all models side by side.
    """
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix = cm,
            display_labels   = ["Typical", "Atypical"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{res['name']}\n"
            f"AUC={res['mean_auc']:.3f} | BalAcc={res['mean_bacc']:.3f}",
            fontsize=11
        )

    plt.suptitle(
        "Confusion Matrices — ML Classifiers\n"
        "Typical vs Atypical Social Communication",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrices saved → {save_path}")


def save_summary(results, save_path):
    """Saves a CSV summary of all model metrics."""
    rows = []
    for res in results:
        rows.append({
            "Model"        : res["name"],
            "Mean AUC"     : round(res["mean_auc"], 4),
            "Std AUC"      : round(res["std_auc"], 4),
            "Mean BalAcc"  : round(res["mean_bacc"], 4),
            "Std BalAcc"   : round(res["std_bacc"], 4),
        })
    df = pd.DataFrame(rows).sort_values("Mean AUC", ascending=False)
    df.to_csv(save_path, index=False)
    print(f"Summary CSV saved → {save_path}")
    print(f"\n{df.to_string(index=False)}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("="*60)
    print("ML Classifiers: RF / SVM / XGBoost")
    print("Pooling: STREAMING ATTENTION (no full RAM load)")
    print("Social Communication Classification from Variant Embeddings")
    print("="*60)
    print(f"Device: {DEVICE}")

    # ── load labels ──
    label_dict = load_labels(CSV_PATH, SCORE_THRESH)

    # ── build lightweight index (paths + labels only, no embeddings) ──
    pairs, y, embed_dim = build_index(PARQUET_DIR, label_dict)

    if len(pairs) == 0:
        raise RuntimeError(
            "No subjects indexed. Check PARQUET_DIR and CSV_PATH."
        )

    # ── run all models ──
    models  = get_models()
    results = []

    for model_name, pipeline in models.items():
        res = run_cv_evaluation(
            model_name, pipeline, pairs, y, embed_dim, N_FOLDS
        )
        results.append(res)

    # ── final comparison ──
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    for res in sorted(results, key=lambda r: r["mean_auc"], reverse=True):
        print(f"  {res['name']:<20} "
              f"AUC={res['mean_auc']:.3f} ± {res['std_auc']:.3f}  "
              f"BalAcc={res['mean_bacc']:.3f} ± {res['std_bacc']:.3f}")

    winner = max(results, key=lambda r: r["mean_auc"])
    print(f"\n  → Best model: {winner['name']} (AUC={winner['mean_auc']:.3f})")

    # ── save outputs ──
    plot_roc_curves(
        results,
        save_path=os.path.join(OUT_DIR, "roc_curves_ml_attn.png")
    )
    plot_confusion_matrices(
        results,
        save_path=os.path.join(OUT_DIR, "confusion_matrices_ml_attn.png")
    )
    save_summary(
        results,
        save_path=os.path.join(OUT_DIR, "ml_results_summary_attn.csv")
    )

    print(f"\nAll outputs saved to: {OUT_DIR}")