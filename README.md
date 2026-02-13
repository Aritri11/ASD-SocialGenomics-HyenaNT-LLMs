# ASD-SocialGenomics-HyenaNT-LLMs

Probing and fine-tuning large foundation LLMs—HyenaDNA/Hyena Transformers and Nucleotide Transformers—to model the genomics of social interaction in Autism Spectrum Disorder (ASD). This project converts variant-level data (VCF) to sequence-level representations (FASTA), generates rich genomic embeddings using these pretrained models, and builds transformer-based classifiers to distinguish individuals with social communication problems related to ASD from controls, aiming to support earlier, scalable intervention beyond traditional ADOS/ADI assessments.

---

## 🔍 Motivation

Autism Spectrum Disorder is highly heterogeneous, involving both rare and common variants that shape social communication and interaction.By learning representations directly from genomic sequences, large genomic LLMs such as HyenaDNA and Nucleotide Transformer can capture long-range regulatory and sequence context that may underlie social behavior phenotypes. This repository explores whether these embeddings can predict individual-level social communication outcomes (e.g., SCQ-like measures) to complement or reduce reliance on time-consuming clinical instruments.

---

## 🧠 Core Idea

- Use **HyenaDNA / Hyena-based Transformers** for long-range genomic modeling (up to very long contexts at single-nucleotide resolution).[web:5]
- Use **Nucleotide Transformer** models as large-scale foundation LLMs trained on human genomics and transcriptomics.
- Generate embeddings from these models for each individual’s genomic data and train downstream transformer classifiers for:
  - ASD vs. non-ASD (where available)
  - Social communication impairment vs. typical social communication (SCQ-like labels or related phenotypes)

---

## 📁 Repository Structure

```text
ASD-SocialGenomics-HyenaNT-LLMs/
├── vcf_to_fasta.py                 # VCF → FASTA conversion pipeline
├── NT_variants.py      # Nucleotide Transformer embedding generation
├── hyena_variants.py   # HyenaDNA/Hyena Transformer embedding generation
├── README.md
├── requirements.txt                # Python dependencies
```
Usage
1. Convert VCF to FASTA
```
python vcf_to_fasta.py \
--vcf_dir /home/group_shyam01/data/SPARK/pub/iWES_V2/variants/deepvariant/gvcf \ #path to the directory for the VCF files
--ref /mnt/data/shyam/aritri/scripts/hg38.fa \ #path to the reference genome 
--out_dir /mnt/data/shyam/aritri/outputs \ #output storage path
--workers 4 \
--shards 4
```

2. Generate Nucleotide Transformer embeddings
```
python nt_variants.py \
  --fasta_dirs data/ \ #fasta files
  --out_dir out/ \ #output files
  --model_name InstaDeepAI/nucleotide-transformer-500m-1000g #nucleotide transformer model in use

```

3. Generate HyenaDNA / Hyena Transformer embeddings
```
python hyena_variants.py \
  --fasta_dirs data/ \ #fasta files
  --out_dir out/ \ #output files
  --model_name LongSafari/hyenadna-small-32k-seqlen #hyena transformer model in use

```

4. Downstream classification (planned)
A separate training script (e.g., train_classifier.py) can consume nt_embeddings.pt and hyena_embeddings.pt along with phenotype labels (e.g., SCQ scores, ASD diagnosis) to train a transformer-based classifier.
This component is intended as a next step and may not yet be included.
