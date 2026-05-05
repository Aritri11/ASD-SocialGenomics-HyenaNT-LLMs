"""
Corrupted Parquet File Scanner
Finds all corrupted or unreadable parquet files in a directory.
Uses a per-file timeout to catch files that hang during reading.

Run:
    python scan_corrupted_parquet.py

Output:
    - Prints corrupted files to console
    - Saves full list to corrupted_files.txt
"""

import os
import signal
import warnings
import pyarrow.parquet as pq
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PARQUET_DIR   = "/mnt/data/shyam/aritri/scripts/out_embed_no_flank"
OUTPUT_FILE   = "/mnt/data/shyam/aritri/scripts/corrupted_files.txt"
TIMEOUT_SEC   = 30     # seconds before a file is considered hanging/corrupted
EMBEDDING_COL = "embedding"  # column to test-read


# ─────────────────────────────────────────────
# TIMEOUT HANDLER
# ─────────────────────────────────────────────

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Read timed out")


# ─────────────────────────────────────────────
# PER-FILE CHECK
# ─────────────────────────────────────────────

def check_file(path, timeout_sec=TIMEOUT_SEC):
    """
    Returns (is_ok, reason) for a single parquet file.
    Checks:
      1. Metadata readable (fast check)
      2. Actual data readable within timeout (catches hangs)
      3. Embedding column exists
      4. At least one valid embedding row
    """
    # ── check 1: metadata ──
    try:
        meta = pq.read_metadata(path)
        if meta.num_rows == 0:
            return False, "empty file (0 rows)"
    except Exception as e:
        return False, f"metadata error: {e}"

    # ── check 2 + 3 + 4: actual read with timeout ──
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        table = pq.read_table(path, columns=[EMBEDDING_COL], memory_map=True)
        signal.alarm(0)   # cancel alarm

        df = table.to_pandas()

        if df.empty:
            return False, "dataframe is empty after read"

        if EMBEDDING_COL not in df.columns:
            return False, f"missing column '{EMBEDDING_COL}'"

        # check at least one valid embedding
        first = df[EMBEDDING_COL].iloc[0]
        if first is None or (hasattr(first, '__len__') and len(first) == 0):
            return False, "first embedding is None or empty"

        return True, "ok"

    except TimeoutError:
        signal.alarm(0)
        return False, f"read timed out after {timeout_sec}s (file may be hanging)"
    except Exception as e:
        signal.alarm(0)
        return False, f"read error: {e}"


# ─────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────

def scan(parquet_dir, output_file):
    files = sorted([
        f for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    ])

    total     = len(files)
    corrupted = []
    ok_count  = 0

    print(f"{'='*60}")
    print(f"Scanning {total} parquet files in:")
    print(f"  {parquet_dir}")
    print(f"Timeout per file: {TIMEOUT_SEC}s")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for i, fname in enumerate(files):
        path     = os.path.join(parquet_dir, fname)
        is_ok, reason = check_file(path)

        if is_ok:
            ok_count += 1
        else:
            corrupted.append((fname, reason))
            print(f"  [{i+1:5d}/{total}] ❌ {fname}")
            print(f"           Reason: {reason}")

        # progress every 500 files
        if (i + 1) % 500 == 0:
            pct = (i + 1) / total * 100
            print(f"\n  --- Progress: {i+1}/{total} ({pct:.1f}%) "
                  f"| OK: {ok_count} | Bad: {len(corrupted)} ---\n")

    # ── summary ──
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total files  : {total}")
    print(f"  OK           : {ok_count}")
    print(f"  Corrupted    : {len(corrupted)}")
    print(f"{'='*60}\n")

    if corrupted:
        print("Corrupted files:")
        for fname, reason in corrupted:
            print(f"  {fname}  [{reason}]")

    # ── save to file ──
    with open(output_file, "w") as f:
        f.write(f"# Corrupted parquet files scan\n")
        f.write(f"# Directory: {parquet_dir}\n")
        f.write(f"# Scanned: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {total} | OK: {ok_count} | Corrupted: {len(corrupted)}\n")
        f.write(f"#\n")
        f.write(f"# Format: filename | reason\n")
        f.write(f"{'#'*60}\n\n")
        for fname, reason in corrupted:
            f.write(f"{fname} | {reason}\n")

    print(f"\nFull list saved → {output_file}")
    return corrupted


if __name__ == "__main__":
    corrupted = scan(PARQUET_DIR, OUTPUT_FILE)

    # print just the filenames at the end for easy copy-paste
    if corrupted:
        print(f"\n{'='*60}")
        print("Corrupted filenames only (for copy-paste):")
        print(f"{'='*60}")
        for fname, _ in corrupted:
            print(fname)