"""
Approach B: Supervised Autoencoder
- Loads mean-pooled features from .npz (cached)
- Trains supervised AE: reconstruction + classification loss
- Uses encoder output for SVM/XGBoost
- 10-fold CV + ROC + confusion matrix plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUT_DIR         = "/mnt/data/shyam/aritri/scripts/ml_results"
FEATURE_CACHE   = os.path.join(OUT_DIR, "mean_pooled_features.npz")
RANDOM_SEED     = 42
N_FOLDS         = 10
N_JOBS          = 8
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Supervised AE config
LATENT_DIM      = 128
EPOCHS          = 40
BATCH_SIZE      = 256
LR              = 1e-3
ALPHA_RECON     = 1.0   # reconstruction weight
BETA_CLASS      = 1.0   # classification weight

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD CACHED FEATURES
# ─────────────────────────────────────────────
def load_features(path):
    data = np.load(path)
    print(f"Loaded cached features → {path}")
    return data["X"], data["y"]

# ─────────────────────────────────────────────
# 2. SUPERVISED AUTOENCODER
# ─────────────────────────────────────────────
class SupervisedAutoencoder(nn.Module):
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
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        logits = self.classifier(z)
        return x_recon, logits, z

def train_supervised_ae(X, y, input_dim, latent_dim):
    print(f"\nTraining Supervised Autoencoder (latent_dim={latent_dim})...")
    model = SupervisedAutoencoder(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    recon_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            x_recon, logits, _ = model(xb)

            recon_loss = recon_loss_fn(x_recon, xb)
            class_loss = class_loss_fn(logits, yb)
            loss = ALPHA_RECON * recon_loss + BETA_CLASS * class_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")
    return model

def encode_features(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        _, _, Z = model(X_tensor)
        return Z.cpu().numpy()

# ─────────────────────────────────────────────
# 3. EVALUATION + PLOTS
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
    colors = {"SVM (SAE features)": "darkorange", "XGBoost (SAE features)": "seagreen"}
    fig, ax = plt.subplots(figsize=(8, 7))
    for res in results:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=colors.get(res["name"], "gray"),
                label=f"{res['name']} (AUC={res['mean_auc']:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"k--",label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Supervised AE, 10‑fold CV)")
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
# 4. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    X_all, y = load_features(FEATURE_CACHE)
    input_dim = X_all.shape[1]
    print(f"Feature matrix shape: {X_all.shape}")

    # Train supervised AE
    sae = train_supervised_ae(X_all, y, input_dim=input_dim, latent_dim=LATENT_DIM)

    # Encode features
    X_encoded = encode_features(sae, X_all)
    print(f"Encoded feature shape: {X_encoded.shape}")

    results = []

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", probability=True,
                   random_state=RANDOM_SEED))
    ])
    results.append(run_cv_evaluation("SVM (SAE features)", svm, X_encoded, y))

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
        results.append(run_cv_evaluation("XGBoost (SAE features)", xgb, X_encoded, y))

    plot_roc_curves(results, os.path.join(OUT_DIR, "roc_curves_sae_10fold.png"))
    plot_confusion_matrices(results, os.path.join(OUT_DIR, "confusion_matrices_sae_10fold.png"))

    print("\nDone.")