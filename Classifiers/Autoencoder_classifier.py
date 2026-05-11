"""
Approach A: Unsupervised Autoencoder for Feature Learning
- Mean pooling of variant embeddings
- Train autoencoder on pooled embeddings
- Use encoder output as input to SVM/XGBoost

NEW:
1. Save mean-pooled features for reuse
2. Plot ROC + confusion matrices and save PNGs
3. 10-fold CV
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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
PARQUET_DIR   = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
CSV_PATH      = "/mnt/data/shyam/aritri/scripts/wes_scq_merged.csv"
OUT_DIR       = "/mnt/data/shyam/aritri/scripts/ml_results"
SCORE_THRESH  = 15
EMBEDDING_COL = "embedding"
MAX_VARIANTS  = None
RANDOM_SEED   = 42
N_FOLDS       = 10
N_JOBS        = 16
N_WORKERS     = 16
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Autoencoder config
AE_LATENT_DIM = 128
AE_EPOCHS     = 30
AE_BATCH_SIZE = 256
AE_LR         = 1e-3

FEATURE_CACHE = os.path.join(OUT_DIR, "mean_pooled_features.npz")

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def extract_subject_id(filename: str) -> str:
    m = re.match(r"(SP\d+)", os.path.basename(filename))
    return m.group(1) if m else None

def load_labels(csv_path: str, threshold: int) -> dict:
    df = pd.read_csv(csv_path)
    df["label"] = (df["summary_score"] > threshold).astype(int)
    label_dict = dict(zip(df["subject_sp_id"].astype(str), df["label"]))
    print(f"Labels loaded: {len(label_dict)} subjects from CSV")
    return label_dict

def is_valid_parquet(path: str) -> bool:
    try:
        meta = pq.read_metadata(path)
        return meta.num_rows > 0
    except Exception:
        return False

def load_raw_embeddings(parquet_path: str, max_variants=None, retries=3) -> np.ndarray:
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
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to read {parquet_path} after {retries} retries: {e}")

def compute_mean_pooled(parquet_path: str, max_variants=None) -> np.ndarray:
    emb = load_raw_embeddings(parquet_path, max_variants)
    return emb.mean(axis=0)

def extract_single_feature(path_label_pair):
    path, _ = path_label_pair
    try:
        pooled = compute_mean_pooled(path, MAX_VARIANTS)
        return pooled, None
    except Exception as e:
        return None, str(e)

def extract_mean_features_parallel(path_label_pairs, n_workers=N_WORKERS):
    features, valid_indices, skipped = [], [], 0
    print(f"  Extracting features with {n_workers} workers (parallel)...")
    with Pool(n_workers) as pool:
        results = tqdm(
            pool.imap_unordered(extract_single_feature, path_label_pairs),
            total=len(path_label_pairs),
            desc="  Extracting features"
        )
        for idx, (feature, error) in enumerate(results):
            if feature is not None:
                features.append(feature)
                valid_indices.append(idx)
            else:
                skipped += 1
    return np.array(features, dtype=np.float32), valid_indices, skipped


def validate_file(file_tuple):
    """Helper function for parallel validation."""
    fname, parquet_dir, label_dict = file_tuple
    sid = extract_subject_id(fname)
    if sid is None or sid not in label_dict:
        return None, "no_label"
    path = os.path.join(parquet_dir, fname)
    if not is_valid_parquet(path):
        return None, "corrupted"
    return (path, label_dict[sid]), "valid"


def build_index(parquet_dir: str, label_dict: dict):
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    print(f"\nFound {len(files)} parquet files")
    print("Validating files during indexing (parallel)...")

    # Prepare tuples for parallel processing
    file_tuples = [(f, parquet_dir, label_dict) for f in files]

    pairs, skipped, corrupt = [], 0, 0

    with Pool(N_WORKERS) as pool:
        results = tqdm(
            pool.imap_unordered(validate_file, file_tuples),
            total=len(files),
            desc="Indexing + validating"
        )
        for pair, status in results:
            if status == "valid":
                pairs.append(pair)
            elif status == "no_label":
                skipped += 1
            elif status == "corrupted":
                corrupt += 1

    labels = np.array([p[1] for p in pairs])
    print(f"Indexed {len(pairs)} valid subjects")
    print(f"  Skipped (no label match) : {skipped}")
    print(f"  Skipped (corrupted)      : {corrupt}")
    print(f"Class 0 (typical)          : {(labels == 0).sum()}")
    print(f"Class 1 (atypical)         : {(labels == 1).sum()}")

    sample = load_raw_embeddings(pairs[0][0], max_variants=1)
    embed_dim = sample.shape[1]
    return pairs, labels, embed_dim

# ─────────────────────────────────────────────
# 2. FEATURE CACHE HELPERS
# ─────────────────────────────────────────────
def save_features(path, X, y):
    np.savez_compressed(path, X=X, y=y)
    print(f"Saved mean-pooled features → {path}")

def load_features(path):
    data = np.load(path)
    print(f"Loaded cached features → {path}")
    return data["X"], data["y"]

# ─────────────────────────────────────────────
# 3. AUTOENCODER
# ─────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def train_autoencoder(X, input_dim, latent_dim):
    print(f"\nTraining Autoencoder (latent_dim={latent_dim})...")
    model = Autoencoder(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, AE_EPOCHS + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{AE_EPOCHS} | Loss: {total_loss/len(loader):.6f}")
    return model

def encode_features(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(DEVICE)
        _, Z = model(X_tensor)
        return Z.cpu().numpy()

# ─────────────────────────────────────────────
# 4. MODEL EVALUATION + PLOTTING
# ─────────────────────────────────────────────
def run_cv_evaluation(name, pipeline, X, y, n_splits=N_FOLDS):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    indices = np.arange(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs, fold_baccs = [], []
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, y)):
        print(f"\n  ── Fold {fold+1}/{n_splits} ──")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, y_prob)
        bacc = balanced_accuracy_score(y_te, y_pred)
        fold_aucs.append(auc)
        fold_baccs.append(bacc)

        all_y_true.extend(y_te)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1}: AUC={auc:.3f}  BalAcc={bacc:.3f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_bacc = np.mean(fold_baccs)
    std_bacc = np.std(fold_baccs)

    print(f"\n  Mean AUC    : {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  Mean BalAcc : {mean_bacc:.3f} ± {std_bacc:.3f}")
    print(classification_report(
        np.array(all_y_true), np.array(all_y_pred),
        target_names=["Typical (<=15)", "Atypical (>15)"]
    ))

    return {
        "name": name,
        "y_true": np.array(all_y_true),
        "y_prob": np.array(all_y_prob),
        "y_pred": np.array(all_y_pred),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_bacc": mean_bacc,
        "std_bacc": std_bacc,
    }

def plot_roc_curves(results, save_path):
    colors = {"SVM (AE features)": "darkorange", "XGBoost (AE features)": "seagreen"}
    fig, ax = plt.subplots(figsize=(8, 7))
    for res in results:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=colors.get(res["name"], "gray"),
                label=f"{res['name']} (AUC={res['mean_auc']:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"k--",label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (10‑fold CV)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC saved → {save_path}")

def plot_confusion_matrices(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Typical","Atypical"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(res["name"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrices saved → {save_path}")

# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("="*60)
    print("Autoencoder Feature Learning (Approach A)")
    print("="*60)
    print(f"Device: {DEVICE}")

    label_dict = load_labels(CSV_PATH, SCORE_THRESH)
    pairs, y, embed_dim = build_index(PARQUET_DIR, label_dict)

    # ---- FEATURE EXTRACTION (CACHED) ----
    if os.path.exists(FEATURE_CACHE):
        X_all, y = load_features(FEATURE_CACHE)
    else:
        print("\nExtracting features (mean pooling, parallel)...")
        X_all, valid_idx, skipped = extract_mean_features_parallel(pairs, n_workers=N_WORKERS)
        y = y[valid_idx]
        save_features(FEATURE_CACHE, X_all, y)

    print(f"Feature matrix shape: {X_all.shape}")

    # Train autoencoder
    ae = train_autoencoder(X_all, input_dim=embed_dim, latent_dim=AE_LATENT_DIM)

    # Encode features
    X_encoded = encode_features(ae, X_all)
    print(f"Encoded feature shape: {X_encoded.shape}")

    results = []

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", probability=True,
                   random_state=RANDOM_SEED))
    ])
    results.append(run_cv_evaluation("SVM (AE features)", svm, X_encoded, y))

    if XGBOOST_AVAILABLE:
        xgb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                scale_pos_weight=1, n_jobs=N_JOBS, random_state=RANDOM_SEED
            ))
        ])
        results.append(run_cv_evaluation("XGBoost (AE features)", xgb, X_encoded, y))

    # Save plots
    plot_roc_curves(results, os.path.join(OUT_DIR, "roc_curves_ae_10fold.png"))
    plot_confusion_matrices(results, os.path.join(OUT_DIR, "confusion_matrices_ae_10fold.png"))

    print("\nDone.")