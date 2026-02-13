#!/usr/bin/env python3
"""
hyena_variants.py (INTRA-SAMPLE parallel)

Each sample is processed by ALL GPUs in parallel.
Each GPU embeds a disjoint subset of variants from the same sample.
"""

import os
import glob
import argparse
import logging
import threading
from queue import Queue
import multiprocessing as mp
from typing import Dict, List
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys

sys.stdout.reconfigure(line_buffering=True)

# ---------------- FASTA utilities ----------------

def iter_fasta_entries(fasta_path):
    with open(fasta_path, "r") as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip().upper())
        if header is not None:
            yield header, "".join(seq_lines)


def parse_header(header: str) -> Dict:
    parts = header.split("|")
    out = {"header": header}

    if len(parts) >= 1:
        out["sample"] = parts[0]

    if len(parts) >= 2 and ":" in parts[1]:
        chrom, pos = parts[1].split(":", 1)
        out["chrom"] = chrom
        out["pos_marker"] = pos

    if len(parts) >= 3 and "-" in parts[2]:
        try:
            s, e = parts[2].split("-", 1)
            out["window_start"] = int(s)
            out["window_end"] = int(e)
        except Exception:
            pass

    for p in parts[3:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v

    return out


# ----------- Timer Utility ------------

class Timer:
    def __init__(self):
        self.t = {}

    def start(self, k):
        self.t[k] = -perf_counter()

    def stop(self, k):
        self.t[k] += perf_counter()

    def get(self, k):
        return self.t.get(k, 0.0)


# ---------------- GPU worker ----------------

def gpu_worker(
    gpu_idx: int,
    indices: np.ndarray,
    sequences: List[str],
    headers: List[str],
    metas: List[Dict],
    args,
    result_queue: mp.Queue,
):
    torch.cuda.set_device(gpu_idx)
    device = f"cuda:{gpu_idx}"

    token_time = 0.0
    gpu_time = 0.0

    # NUMA pinning
    if gpu_idx < 2:
        os.sched_setaffinity(0, range(0, 64))
    else:
        os.sched_setaffinity(0, range(64, 128))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).half().eval()

    batch_size = args.batch_size
    max_len = tokenizer.model_max_length

    token_queue = Queue(maxsize=8)  # FIX 4
    STOP = object()

    local_rows = []

    def tokenizer_thread():
        nonlocal token_time
        batch_texts, batch_ids = [], []

        for idx in indices:
            batch_texts.append(sequences[idx])
            batch_ids.append(idx)

            if len(batch_texts) == batch_size:
                t0 = perf_counter()
                tokens = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=max_len,
                )
                token_time += perf_counter() - t0
                token_queue.put((tokens, batch_ids))
                batch_texts, batch_ids = [], []

        if batch_texts:
            t0 = perf_counter()
            tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_len,
            )
            token_time += perf_counter() - t0
            token_queue.put((tokens, batch_ids))

        token_queue.put(STOP)

    t = threading.Thread(target=tokenizer_thread, daemon=True)
    t.start()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        with tqdm(
            total=len(indices),
            desc=f"[GPU-{gpu_idx}] variants",
            position=gpu_idx + 1,
            leave=False,
            unit="variant",
        ) as pbar:

            while True:
                item = token_queue.get()
                if item is STOP:
                    break

                tokens, batch_ids = item
                tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}

                t0 = perf_counter()
                out = model(**tokens).last_hidden_state.mean(dim=1)
                torch.cuda.synchronize()
                gpu_time += perf_counter() - t0

                out = out.cpu().numpy()

                for i, emb in zip(batch_ids, out):
                    meta = metas[i]
                    local_rows.append({
                        "index_in_sample": i,
                        "header": headers[i],
                        "sample": meta.get("sample"),
                        "chrom": meta.get("chrom"),
                        "pos_marker": meta.get("pos_marker"),
                        "window_start": meta.get("window_start"),
                        "window_end": meta.get("window_end"),
                        "reflen": meta.get("reflen"),
                        "allele": meta.get("allele"),
                        "embedding": emb.astype(np.float32),
                    })

                pbar.update(len(batch_ids))

    # FIX 1 + 2: single, final queue write
    result_queue.put({
        "rows": local_rows,
        "token_time": token_time,
        "gpu_time": gpu_time,
    })


# ---------------- Main ----------------

def main():
    print(">>> Script started")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fasta_files = []
    for d in args.fasta_dirs:
        fasta_files.extend(glob.glob(os.path.join(d, "*.fa")))
    fasta_files = sorted(fasta_files)

    num_gpus = torch.cuda.device_count()
    ctx = mp.get_context("spawn")

    with tqdm(fasta_files, desc="Embedding samples", unit="sample") as sample_pbar:
        for fasta_path in sample_pbar:
            sample_id = os.path.basename(fasta_path).replace(".fa", "")
            out_path = os.path.join(args.out_dir, f"{sample_id}.parquet")

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue

            headers, metas, sequences = [], [], []
            for h, s in iter_fasta_entries(fasta_path):
                headers.append(h)
                metas.append(parse_header(h))
                sequences.append(s)

            indices_split = np.array_split(np.arange(len(sequences)), num_gpus)

            result_queue = ctx.Queue()
            processes = []

            for gpu_idx in range(num_gpus):
                if len(indices_split[gpu_idx]) == 0:
                    continue
                p = ctx.Process(
                    target=gpu_worker,
                    args=(
                        gpu_idx,
                        indices_split[gpu_idx],
                        sequences,
                        headers,
                        metas,
                        args,
                        result_queue,
                    ),
                )
                p.start()
                processes.append(p)

            all_rows = []
            total_token_time = 0.0
            total_gpu_time = 0.0

            for _ in processes:
                res = result_queue.get()
                all_rows.extend(res["rows"])
                total_token_time += res["token_time"]
                total_gpu_time += res["gpu_time"]

            for p in processes:
                p.join()

            all_rows.sort(key=lambda x: x["index_in_sample"])

            table = pa.Table.from_pylist(
                all_rows,
                schema=pa.schema({
                    "index_in_sample": pa.int32(),
                    "header": pa.string(),
                    "sample": pa.string(),
                    "chrom": pa.string(),
                    "pos_marker": pa.string(),
                    "window_start": pa.int32(),
                    "window_end": pa.int32(),
                    "reflen": pa.string(),
                    "allele": pa.string(),
                    "embedding": pa.list_(pa.float32()),
                }),
            )

            pq.write_table(table, out_path)

    print("All done.")


if __name__ == "__main__":
    main()
