# Run this small Python script to check which parquets are corrupted
import pyarrow.parquet as pq
import glob

out_dir = "/mnt/data/shyam/aritri/scripts/out_embed"
for f in glob.glob(out_dir + "*.parquet"):
    try:
        t = pq.read_table(f)
        print(f"OK ({len(t)} rows): {f}")
    except Exception as e:
        print(f"CORRUPT: {f} -> {e}")