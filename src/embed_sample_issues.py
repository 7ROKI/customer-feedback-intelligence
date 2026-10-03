import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "data/cleaned_reviews.csv"

OUTPUT_CSV = "data/sample_issue_embeddings.csv"
OUTPUT_NPY = "data/sample_issue_embeddings.npy"

SAMPLE_SIZE = 40


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print("Total cleaned reviews:", len(df))


# ============================================================
# 3. Take a reproducible sample
# ============================================================

sample_df = df.sample(
    n=SAMPLE_SIZE,
    random_state=42
).copy()

sample_df = sample_df.reset_index(drop=True)

print("Sample size:", len(sample_df))


# ============================================================
# 4. Load BGE-M3
# ============================================================

print("\nLoading BGE-M3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)


# ============================================================
# 5. Generate embeddings
# ============================================================

texts = sample_df["feedback"].tolist()

print("\nGenerating embeddings...")

output = model.encode(
    texts,
    batch_size=4,
    max_length=128
)

embeddings = output["dense_vecs"]

print("Embedding shape:", embeddings.shape)


# ============================================================
# 6. Save embeddings
# ============================================================

np.save(
    OUTPUT_NPY,
    embeddings
)

sample_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nFiles saved:")
print(OUTPUT_CSV)
print(OUTPUT_NPY)

print("\nSample embedding generation completed!")