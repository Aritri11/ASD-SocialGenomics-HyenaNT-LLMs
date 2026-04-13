"""
Vanilla Transformer (sin/cos positional encoding) with
ATTENTION POOLING for sample-level classification
from variant embeddings stored in parquet files.
"""

import os
import glob
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler


def setup_ddp():
    print(f"[BEFORE INIT] PID={os.getpid()} RANK={os.environ.get('RANK')}", flush=True)
    dist.init_process_group(backend="nccl")
    print(f"[AFTER INIT]  PID={os.getpid()} RANK={os.environ.get('RANK')}", flush=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def destroy_ddp():
    dist.destroy_process_group()


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (B,) or (B, num_classes) - model output
            targets: (B,) - ground truth labels
        """
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce).mean()
        return focal_loss


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, positions):
        """
        x: (B, L, D)
        positions: (B, L) genomic coordinates
        """
        device = x.device
        positions = positions.float().unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(positions * div_term)
        pe[..., 1::2] = torch.cos(positions * div_term)
        return x + pe


# Attention Pooling
class AttentionPooling(nn.Module):
    """
    Learns to weight variants instead of mean-pooling them.
    """

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        weights = torch.softmax(self.attn(x), dim=1)  # (B, L, 1)
        return (weights * x).sum(dim=1)  # (B, D)


# Dataset
CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHR_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}

CHR_OFFSET = {}
offset = 0
for chrom in CHR_ORDER:
    CHR_OFFSET[chrom] = offset
    offset += CHR_LENGTHS[chrom]
CHR_TO_IDX = {c: i for i, c in enumerate(CHR_ORDER)}


class VariantParquetDataset(Dataset):
    def __init__(self, parquet_dir, labels_csv, max_variants=999999):
        all_files = sorted(glob.glob(os.path.join(parquet_dir, "*.gvcf_variants.parquet")))
        print(f"[DEBUG] Glob pattern: {os.path.join(parquet_dir, '*.gvcf_variants.parquet')}")
        print(f"[DEBUG] Files found: {len(all_files)}")
        if all_files:
            print(f"[DEBUG] First file: {all_files[0]}")
        else:
            # Show what IS in the directory
            actual = os.listdir(parquet_dir)[:5]
            print(f"[DEBUG] Actual files in dir: {actual}")
        valid_files = []
        for fp in all_files:
            try:
                pq.read_metadata(fp)
                valid_files.append(fp)
            except (pyarrow.lib.ArrowInvalid, OSError) as e:
                warnings.warn(f"[SKIP] Corrupted parquet file: {fp}")

        self.files = valid_files
        labels_df = pd.read_csv(labels_csv)
        self.labels = dict(zip(labels_df["subject_sp_id"], labels_df["summary_score"]))
        self.max_variants = max_variants

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        subject_sp_id = os.path.basename(fp).replace(".gvcf_variants.parquet", "")

        try:
            table = pq.read_table(fp)
        except (pyarrow.lib.ArrowInvalid, OSError) as e:
            warnings.warn(f"[SKIP] Failed to read: {fp}")
            return None

        df = table.to_pandas()

        df["chrom_idx"] = df["chrom"].map(CHR_TO_IDX)
        df = df.sort_values(["chrom_idx", "window_start"])

        emb_list = []

        for e in df["embedding"].values:
            arr = np.array(e, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 256:
                emb_list.append(arr)
            else:
                continue  # skip corrupted row
        if not emb_list:
            return None

        embeddings = np.stack(emb_list).astype(np.float32)

        genomic_positions = (
                df["chrom"].map(CHR_OFFSET).values
                + df["window_start"].values
        ).astype(np.int64)

        embeddings = embeddings[: self.max_variants]
        positions = genomic_positions[: self.max_variants]

        if subject_sp_id not in self.labels:
            raise KeyError(f"Missing label for subject_sp_id={subject_sp_id}")

        raw_score = int(self.labels[subject_sp_id])
        summary_score = 1 if raw_score >= 15 else 0

        return {
            "embeddings": torch.from_numpy(embeddings),
            "positions": torch.from_numpy(positions),
            "summary_score": torch.tensor(summary_score, dtype=torch.long),
        }


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# Transformer Model (with Attention Pooling)
class VariantTransformer(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            ff_mult=4,
            num_classes=2,
            max_len=50000,
            chunk_size=512,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        # self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = AttentionPooling(embed_dim)

        self.chunk_pool = AttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, embeddings, positions):
        chunk_reprs = []

        for i in range(0, embeddings.size(0), self.chunk_size):
            emb = embeddings[i: i + self.chunk_size].unsqueeze(0)  # (1, C, D)
            pos = positions[i: i + self.chunk_size].unsqueeze(0)  # (1, C)

            x = self.pos_encoder(emb, pos)
            x = self.encoder(x)

            chunk_repr = self.pool(x)  # (1, D)
            chunk_reprs.append(chunk_repr)

        # Stack chunk representation
        # chunk reprs: list of (1, D)
        chunk_tensor = torch.cat(chunk_reprs, dim=0).unsqueeze(0)
        # shape: (1, num_chunks, D)

        # Learn importance of chunks
        sample_repr = self.chunk_pool(chunk_tensor)  # (1, D)

        logits = self.classifier(sample_repr.squeeze(0))
        return logits


# Label distribution
def print_label_distribution(labels_csv, threshold=15):
    labels_df = pd.read_csv(labels_csv)

    raw_scores = labels_df["summary_score"].values
    binary = (raw_scores >= threshold).astype(int)

    n_pos = binary.sum()
    n_neg = len(binary) - n_pos

    total = n_pos + n_neg
    print("=" * 50)
    print(f"GROUND TRUTH LABEL DISTRIBUTION")
    print(f"Total Samples:{total}")
    print(f"Positives (1):{n_pos} ({n_pos / total:.3f})")
    print(f"Negatives (0):{n_neg} ({n_neg / total:.3f})")
    print("=" * 50)


# Training loop
def train(args, local_rank):
    device = torch.device(f"cuda:{local_rank}")

    dataset = VariantParquetDataset(args.parquet_dir, args.labels_csv)
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0, pin_memory=True,
                              collate_fn=collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=0, pin_memory=True,
                            collate_fn=collate_skip_none)

    model = VariantTransformer(
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        chunk_size=args.chunk_size,
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    # ✅ Use FocalLoss instead of CrossEntropyLoss
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scaler = GradScaler("cuda")

    for epoch in range(args.epochs):
        # ============================
        # Training
        # ============================
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]"):
            if batch is None:
                continue
            emb = batch["embeddings"].to(device).squeeze(0)
            pos = batch["positions"].to(device).squeeze(0)
            label = batch["summary_score"].to(device)
            assert label.dtype == torch.long, f"Got {label.dtype}"
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(emb, pos)
                # ✅ FocalLoss expects logits shape (B, num_classes) and targets shape (B,)
                loss = criterion(logits.unsqueeze(0), label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if dist.get_rank() == 0:
                print(f"loss={loss.item():.6f} | scale={scaler.get_scale():.1f} | "
                      f"label={label.item()} | pred={logits.argmax().item()}")

            total_loss += loss.item()

        if dist.get_rank() == 0:
            print(
                f"Epoch {epoch + 1} | Train Loss: "
                f"{total_loss / max(1, (len(train_loader)))} | "
            )

        # =========================
        # Validation + Metrics
        # =========================
        model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [val]"):
                if batch is None:
                    continue
                emb = batch["embeddings"].to(device).squeeze(0)
                pos = batch["positions"].to(device).squeeze(0)
                label_tensor = batch["summary_score"].to(device)
                assert label_tensor.dtype == torch.long, f"Got {label_tensor.dtype}"
                label = label_tensor.item()

                logits = model(emb, pos)
                # print(f"[DEBUG] Logits shape inside validation loop: {logits.shape}")  # It is 2 as well.
                probs = torch.softmax(logits, dim=-1)  # changed from 0 to -1

                pred = torch.argmax(probs).item()
                prob_1 = probs[1].item()

                y_true.append(label)
                y_pred.append(pred)
                y_prob.append(prob_1)

        if dist.get_rank() == 0:
            if not y_true:
                print(f"[WARNING] Epoch {epoch + 1}: validation set was empty, skipping metrics.")
                continue
            print(
                f"Logits Min:{logits.min().item()}",
                f"Logits Max:{logits.max().item()}",
                f"Logits Mean:{logits.mean().item()}",
            )

            TP = TN = FP = FN = 0
            for yt, yp in zip(y_true, y_pred):
                if yt == 0 and yp == 0:
                    TN += 1
                elif yt == 1 and yp == 1:
                    TP += 1
                elif yt == 0 and yp == 1:
                    FP += 1
                elif yt == 1 and yp == 0:
                    FN += 1
            print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")

            print(
                f"[Epoch {epoch + 1} | VAL] "
                f"Acc={acc:.4f} | Prec={prec:.4f} | "
                f"Rec={rec:.4f} | F1={f1:.4f} | AUROC={auc:.4f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(args.out_dir, "transformer_variant_classifier.pt"),
    )


# =====================================================
# CLI
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir")
    parser.add_argument("--parquet_dir")
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    local_rank = setup_ddp()
    try:
        train(args, local_rank)
    finally:
        dist.destroy_process_group()  # this way it destroys dist processes if train ever crashes


if __name__ == "__main__":
    main()