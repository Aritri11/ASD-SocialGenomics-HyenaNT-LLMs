"""
Classical ML Classifiers for Social Communication Classification
Random Forest + SVM + XGBoost on genomic variant embeddings

Input:
  - Parquet files: variant embeddings per subject
  - CSV file    : SCQ scores per subject

Classification:
  SCQ score <= 15 → Typical social communication    (class 0)
  SCQ score >  15 → Atypical social communication   (class 1)

Pooling strategy: MEAN POOLING
  Simple average of all variant embeddings per subject.
  Each variant contributes equally to the final subject representation.
  RAM efficient: streams embeddings from disk, computes mean, discards raw data.

Output per model:
  - Classification report (precision, recall, F1)
  - AUC-ROC score
  - ROC curve plot (saved as PNG)

Run:
  python ml_classifiers_mean.py
"""
import time
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

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

import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PARQUET_DIR = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
CSV_PATH = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"
OUT_DIR = "/mnt/data/shyam/aritri/scripts/ml_results"
SCORE_THRESH = 15  # <= 15 → typical (0), > 15 → atypical (1)
EMBEDDING_COL = "embedding"
MAX_VARIANTS = None  # None = use all variants
RANDOM_SEED = 42
N_FOLDS = 5  # stratified K-fold CV
N_JOBS = 8  # CPU cores for RF and XGBoost

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    label_dict = dict(zip(df["subject_sp_id"].astype(str), df["label"]))
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



def load_raw_embeddings(parquet_path: str, max_variants=None, retries=3) -> np.ndarray:
    """Returns (N_variants, embed_dim) float32 array with retry logic."""
    for attempt in range(retries):
        try:
            df  = pd.read_parquet(parquet_path, columns=[EMBEDDING_COL])
            emb = np.array(df[EMBEDDING_COL].tolist(), dtype=np.float32)
            if max_variants and len(emb) > max_variants:
                idx = np.random.choice(len(emb), max_variants, replace=False)
                emb = emb[idx]
            return emb
        except OSError as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"  ⚠ Retry {attempt+1}/{retries} after {wait_time}s (network issue)")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to read {parquet_path} after {retries} retries: {e}")


def compute_mean_pooled(parquet_path: str,
                        max_variants=None) -> np.ndarray:
    """
    Streams one parquet file, computes mean embedding, discards raw data.
    RAM usage: only (embed_dim,) = 1280 floats per subject instead of
               (46000, 1280) = 59M floats.
    """
    emb = load_raw_embeddings(parquet_path, max_variants)
    return emb.mean(axis=0)  # (embed_dim,) — tiny


# ─────────────────────────────────────────────
# 2. MEAN POOLING (SIMPLIFIED)
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_mean_features(path_label_pairs) -> tuple:
    """
    Streams parquet files, skips files with read errors.
    Returns (features, valid_indices, skipped_count).
    """
    features = []
    valid_indices = []
    skipped = 0

    for idx, (path, _) in enumerate(tqdm(path_label_pairs, desc="  Extracting features")):
        try:
            pooled = compute_mean_pooled(path, MAX_VARIANTS)
            features.append(pooled)
            valid_indices.append(idx)
        except Exception as e:
            print(f"\n  ⚠ Skipping {os.path.basename(path)}: {str(e)[:60]}")
            skipped += 1
            continue

    return np.array(features, dtype=np.float32), valid_indices, skipped

def build_index(parquet_dir: str, label_dict: dict):
    """
    Builds a lightweight index of (path, label) pairs.
    Validates each file's metadata during indexing so corrupted
    files are caught NOW — not 12 hours into training.
    RAM usage: negligible (strings only).
    """
    files = sorted([f for f in os.listdir(parquet_dir)
                    if f.endswith(".parquet")])
    print(f"\nFound {len(files)} parquet files")
    print(f"Validating files during indexing (metadata check)...")

    pairs = []
    skipped = 0
    corrupt = 0

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
    sample = load_raw_embeddings(pairs[0][0], max_variants=1)
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
                n_estimators=500,
                max_depth=None,
                class_weight="balanced",  # handles class imbalance
                n_jobs=N_JOBS,
                random_state=RANDOM_SEED,
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",  # handles class imbalance
                probability=True,  # needed for AUC-ROC
                random_state=RANDOM_SEED,
            ))
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=1,  # adjusted below per fold
                n_jobs=N_JOBS,
                random_state=RANDOM_SEED,
            ))
        ])
    else:
        print("⚠ XGBoost not available — skipping. Install with: pip install xgboost")

    return models


# ─────────────────────────────────────────────
# 4. EVALUATION (CV with mean pooling)
# ─────────────────────────────────────────────

def run_cv_evaluation_with_precomputed(name, pipeline, X, y, n_splits=N_FOLDS):
    """
    Stratified K-fold CV with pre-computed features (no extraction).
    Much faster since features extracted only once, not per model.
    """
    print(f"\n{'='*60}")
    print(f"Model: {name}  (pre-computed features)")
    print(f"{'='*60}")

    indices    = np.arange(len(X))
    skf        = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RANDOM_SEED)
    fold_aucs, fold_baccs = [], []
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, y)):
        print(f"\n  ── Fold {fold+1}/{n_splits} ──")

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ── per-fold XGBoost imbalance adjustment ──
        if name == "XGBoost" and XGBOOST_AVAILABLE:
            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            pipeline.named_steps["clf"].set_params(
                scale_pos_weight=n_neg / max(n_pos, 1)
            )

        # ── fit and evaluate ──
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
        "Random Forest": "steelblue",
        "SVM": "darkorange",
        "XGBoost": "seagreen",
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
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Typical", "Atypical"]
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
            "Model": res["name"],
            "Mean AUC": round(res["mean_auc"], 4),
            "Std AUC": round(res["std_auc"], 4),
            "Mean BalAcc": round(res["mean_bacc"], 4),
            "Std BalAcc": round(res["std_bacc"], 4),
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

    print("=" * 60)
    print("ML Classifiers: RF / SVM / XGBoost")
    print("Pooling: STREAMING MEAN (simple average, no RAM load)")
    print("Social Communication Classification from Variant Embeddings")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── load labels ──
    label_dict = load_labels(CSV_PATH, SCORE_THRESH)

    # ── build lightweight index ──
    pairs, y, embed_dim = build_index(PARQUET_DIR, label_dict)

    if len(pairs) == 0:
        raise RuntimeError("No subjects indexed. Check PARQUET_DIR and CSV_PATH.")

    # ── EXTRACT ALL FEATURES ONCE ──
    print(f"\nExtracting features for all subjects...")
    X_all, all_valid_idx, all_skipped = extract_mean_features(pairs)

    # Filter to valid samples only
    pairs = [pairs[i] for i in all_valid_idx]
    y = y[all_valid_idx]

    print(f"Total subjects after filtering: {len(pairs)}")
    print(f"Total subjects skipped: {all_skipped}")

    # ── run all models with pre-computed features ──
    models = get_models()
    results = []

    for model_name, pipeline in models.items():
        res = run_cv_evaluation_with_precomputed(
            model_name, pipeline, X_all, y, N_FOLDS
        )
        results.append(res)

    # ── final comparison ──
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")
    for res in sorted(results, key=lambda r: r["mean_auc"], reverse=True):
        print(f"  {res['name']:<20} "
              f"AUC={res['mean_auc']:.3f} ± {res['std_auc']:.3f}  "
              f"BalAcc={res['mean_bacc']:.3f} ± {res['std_bacc']:.3f}")

    winner = max(results, key=lambda r: r["mean_auc"])
    print(f"\n  → Best model: {winner['name']} (AUC={winner['mean_auc']:.3f})")

    # ── save outputs ──
    plot_roc_curves(results, save_path=os.path.join(OUT_DIR, "roc_curves_ml_mean.png"))
    plot_confusion_matrices(results, save_path=os.path.join(OUT_DIR, "confusion_matrices_ml_mean.png"))
    save_summary(results, save_path=os.path.join(OUT_DIR, "ml_results_summary_mean.csv"))

    print(f"\nAll outputs saved to: {OUT_DIR}")