"""
Hyperparameter tuning + evaluation for SVM and XGBoost
- Loads mean-pooled features from .npz
- GridSearchCV with verbose progress
- Prints classification report
- Saves ROC curve + confusion matrix plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUT_DIR       = "/mnt/data/shyam/aritri/scripts/ml_results"
FEATURE_CACHE = os.path.join(OUT_DIR, "mean_pooled_features.npz")
RANDOM_SEED   = 42
N_JOBS        = 16
N_FOLDS       = 5

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────
data = np.load(FEATURE_CACHE)
X_all, y = data["X"], data["y"]
print(f"Loaded features: {X_all.shape}")

# Use one split for tuning (fast)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
tr_idx, te_idx = next(skf.split(X_all, y))
X_tr, y_tr = X_all[tr_idx], y[tr_idx]

# Standardize once for tuning
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)

# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────
def run_cv_evaluation(name, pipeline, X, y, n_splits=N_FOLDS):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    indices = np.arange(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    all_y_true, all_y_prob, all_y_pred = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(indices, y)):
        print(f"  Fold {fold+1}/{n_splits}")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    auc = roc_auc_score(all_y_true, all_y_prob)
    bacc = balanced_accuracy_score(all_y_true, all_y_pred)

    print(f"\nAUC: {auc:.3f} | Balanced Acc: {bacc:.3f}")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=["Typical (<=15)", "Atypical (>15)"]
    ))

    return {
        "name": name,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
        "auc": auc
    }

def plot_roc(res, save_path):
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{res['name']} (AUC={res['auc']:.3f})")
    plt.plot([0,1],[0,1],"k--",label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {res['name']}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC saved → {save_path}")

def plot_confusion(res, save_path):
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Typical","Atypical"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {res['name']}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")

# ─────────────────────────────────────────────
# SVM TUNING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SVM HYPERPARAMETER TUNING (GridSearchCV)")
print("="*60)

svm_param_grid = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "kernel": ["rbf", "poly"],
}

svm_grid = GridSearchCV(
    estimator=SVC(class_weight="balanced", probability=True, random_state=RANDOM_SEED),
    param_grid=svm_param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=N_JOBS,
    verbose=3
)

svm_grid.fit(X_tr_scaled, y_tr)
print("\nBest SVM params:", svm_grid.best_params_)
print("Best SVM AUC:", svm_grid.best_score_)

svm_best = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", svm_grid.best_estimator_)
])

svm_res = run_cv_evaluation("SVM (Tuned)", svm_best, X_all, y)
plot_roc(svm_res, os.path.join(OUT_DIR, "roc_svm_tuned.png"))
plot_confusion(svm_res, os.path.join(OUT_DIR, "cm_svm_tuned.png"))

# ─────────────────────────────────────────────
# XGBOOST TUNING
# ─────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("\n" + "="*60)
    print("XGBOOST HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*60)

    xgb_param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(
            n_estimators=500,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            verbosity=0
        ),
        param_grid=xgb_param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=N_JOBS,
        verbose=3
    )

    xgb_grid.fit(X_tr_scaled, y_tr)
    print("\nBest XGBoost params:", xgb_grid.best_params_)
    print("Best XGBoost AUC:", xgb_grid.best_score_)

    xgb_best = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb_grid.best_estimator_)
    ])

    xgb_res = run_cv_evaluation("XGBoost (Tuned)", xgb_best, X_all, y)
    plot_roc(xgb_res, os.path.join(OUT_DIR, "roc_xgb_tuned.png"))
    plot_confusion(xgb_res, os.path.join(OUT_DIR, "cm_xgb_tuned.png"))