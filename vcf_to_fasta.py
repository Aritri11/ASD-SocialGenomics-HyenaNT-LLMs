#!/usr/bin/env python3
"""

Produces variant-centered FASTA windows (one FASTA entry per variant) from per-sample VCFs.
Shards output into N directories (for parallel embedding across N GPUs).
"""

import argparse
import os
import glob
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pysam
import logging

def write_fasta_entry(fp, header, seq):
    fp.write(f">{header}\n")
    for i in range(0, len(seq), 60):
        fp.write("".join(seq[i:i+60]) + "\n")


def extract_variant_windows(vcf_file, ref_fasta, out_dir, flank, qual_thresh, dp_thresh,
                            max_windows, seed, shard_id, shards):
    """
    Process one VCF file: extract windows and write FASTA into a shard subdir.
    Returns (sample_name, status, n_windows).
      status: "ok", "skipped_no_tbi", "no_variants", etc.
    """
    basename = os.path.basename(vcf_file)
    sample_name = os.path.splitext(basename)[0]
    out_subdir = os.path.join(out_dir, f"shard_{shard_id}")
    os.makedirs(out_subdir, exist_ok=True)
    out_fpath = os.path.join(out_subdir, f"{sample_name}_variants.fa")

    # require tabix index for random access
    if not os.path.exists(vcf_file + ".tbi"):
        return (sample_name, "skipped_no_tbi", 0)

    try:
        ref = pysam.FastaFile(ref_fasta)
    except Exception as e:
        return (sample_name, f"ref_open_error:{e}", 0)

    try:
        vcf = pysam.VariantFile(vcf_file)
    except Exception as e:
        return (sample_name, f"vcf_open_error:{e}", 0)

    sample = list(vcf.header.samples)[0] if vcf.header.samples else None
    if sample is None:
        return (sample_name, "no_sample_in_vcf", 0)

    windows = []
    # iterate by chromosome to avoid loading whole genome
    for chrom in ref.references:
        chrom_len = ref.get_reference_length(chrom)
        logging.info(f"------------------------------------------------------------------")
        logging.info(f"Chromosome {chrom} Length: {chrom_len}")
        logging.info(f"------------------------------------------------------------------")

        # walk variants in this chrom
        for rec in vcf.fetch(chrom):
            # QUAL filter (some VCFs may have None)
            if rec.qual is None or rec.qual < qual_thresh:
                continue

            # genotype presence
            gt = rec.samples.get(sample, {}).get("GT", None)
            if gt is None or all(g is None for g in gt):
                continue

            # per-sample DP check if present
            sample_dp = rec.samples[sample].get("DP", None)
            if dp_thresh is not None and sample_dp is not None and sample_dp < dp_thresh:
                continue

            # construct window coordinates (0-based)
            start0 = rec.pos - 1
            end0 = start0 + len(rec.ref)
            region_start = max(0, start0 - flank)
            region_end = min(chrom_len, end0 + flank)
            if region_end <= region_start:
                continue

            allele_index = gt[0] if gt[0] is not None else 0
            alleles = [rec.ref] + list(rec.alts or [])
            try:
                allele_seq = alleles[allele_index]
            except Exception:
                allele_seq = rec.ref  # fallback

            windows.append((chrom, rec.pos, region_start, region_end, start0 - region_start, len(rec.ref), allele_seq))

    # no windows found
    if not windows:
        return (sample_name, "no_variants", 0)

    # deterministic subsample if needed
    rng = random.Random(seed + hash(sample_name))
    if max_windows is not None and len(windows) > max_windows:
        windows = rng.sample(windows, max_windows)

    # write fasta
    with open(out_fpath, "w") as out:
        for chrom, pos, rstart, rend, rel_start, ref_len, allele_seq in windows:
            seq = list(ref.fetch(chrom, rstart, rend))
            # replace ref slice with allele sequence in local coords
            rel_end = rel_start + ref_len
            seq[rel_start:rel_end] = list(allele_seq)
            header = f"{sample_name}|{chrom}:{pos}|{rstart+1}-{rend}|reflen={ref_len}|allele={allele_seq}"
            write_fasta_entry(out, header, seq)

    return (sample_name, "ok", len(windows))


def worker_args(i, vcf_file, args):
    # assign shard id round-robin for this vcf
    shard_id = i % args.shards
    return (vcf_file, args.ref, args.out_dir, args.flank, args.qual, args.dp,
            args.max_windows, args.seed, shard_id, args.shards)


def run_extract(arg_tuple):
    return extract_variant_windows(*arg_tuple)


def main():
    parser = argparse.ArgumentParser(description="Create variant-centered FASTA windows and shard outputs.")
    parser.add_argument("--vcf_dir", required=True, help="Directory containing .vcf.gz or .gvcf.gz files")
    parser.add_argument("--ref", default=" /mnt/data/shyam/aritri/scripts/hg38_ref/hg38_primary_allchr.fa", help="Reference FASTA")
    parser.add_argument("--out_dir", required=True, help="Output directory for shard folders")
    parser.add_argument("--flank", required=True, type=int, default=4096, help="Flank size (bp) on each side of variant (default 100)")
    parser.add_argument("--qual", type=float, default=30.0, help="Minimum QUAL filter (default 30)")
    parser.add_argument("--dp", type=int, default=10, help="Minimum per-sample DP (FORMAT/DP). If absent, not applied.")
    parser.add_argument("--max_windows", type=int, default=None, help="Max windows per sample (None for no cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible subsampling")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count()-1), help="Number of CPU workers")
    parser.add_argument("--shards", type=int, default=4, help="Number of shards (e.g., GPUs) to split outputs across")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # gather VCFs
    vcf_files = sorted(
        glob.glob(os.path.join(args.vcf_dir, "*.vcf.gz")) +
        glob.glob(os.path.join(args.vcf_dir, "*.gvcf.gz"))
    )
    if not vcf_files:
        print(f"[ERROR] No VCF/gVCF files found in {args.vcf_dir}")
        return
    print(f"[INFO] Found {len(vcf_files)} VCF/gVCF files to process. Sharding into {args.shards} directories.")

    # prepare worker argument tuples (round-robin shard assignment)
    task_list = [worker_args(i, vcf_file, args) for i, vcf_file in enumerate(vcf_files)]

    results = []
    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(run_extract, task_list), total=len(task_list)):
            results.append(res)

    # summary
    ok = [r for r in results if r[1] == "ok"]
    skipped = [r for r in results if r[1] != "ok"]
    print(f"[INFO] Successfully created FASTA for {len(ok)}/{len(vcf_files)} samples.")
    if skipped:
        print("[INFO] Skipped samples summary:")
        for sname, status, n in skipped:
            print(f"  - {sname}: {status}")


if __name__ == "__main__":
    main()
