import pandas as pd
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "data/all_reviews_structured.csv"

OUTPUT_CSV = "data/specific_issues_for_clustering.csv"
OUTPUT_NPY = "data/specific_issue_embeddings.npy"


# ============================================================
# 2. Load structured reviews
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Total structured reviews:", len(df))


# ============================================================
# 3. Keep only actionable / specific issues
# ============================================================

specific_df = df[
    df["has_specific_issue"]
    .astype(str)
    .str.lower()
    == "true"
].copy()

specific_df = specific_df.reset_index(drop=True)

print(
    "Specific-issue reviews:",
    len(specific_df)
)


# ============================================================
# 4. Load BGE-M3
# ============================================================

print("\nLoading BGE-M3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)


# ============================================================
# 5. Embed the CLEAN specific_issue phrase
#    rather than the whole raw review
# ============================================================

texts = (
    specific_df["specific_issue"]
    .astype(str)
    .tolist()
)

print("\nGenerating embeddings...")

output = model.encode(
    texts,
    batch_size=8,
    max_length=128
)

embeddings = output["dense_vecs"]


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# 6. Save
# ============================================================

specific_df.to_csv(
    OUTPUT_CSV,
    index=False
)

np.save(
    OUTPUT_NPY,
    embeddings
)


print("\nFiles saved:")
print(OUTPUT_CSV)
print(OUTPUT_NPY)

print(
    "\nAll specific-issue embeddings completed!"
)