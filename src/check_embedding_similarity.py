import pandas as pd
import numpy as np


# ============================================================
# 1. Load sample data and embeddings
# ============================================================

CSV_FILE = "data/sample_issue_embeddings.csv"
NPY_FILE = "data/sample_issue_embeddings.npy"

df = pd.read_csv(CSV_FILE, dtype=str)
embeddings = np.load(NPY_FILE)

print("Reviews:", len(df))
print("Embedding shape:", embeddings.shape)


# ============================================================
# 2. Normalize embeddings
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

normalized = embeddings / norms


# ============================================================
# 3. Cosine similarity matrix
# ============================================================

similarity_matrix = np.dot(
    normalized,
    normalized.T
)


# ============================================================
# 4. Show the most similar review for each review
# ============================================================

print("\n=== Most Similar Review Pairs ===")

pairs = []

for i in range(len(df)):

    # Ignore similarity with itself
    similarity_matrix[i, i] = -1

    j = np.argmax(similarity_matrix[i])

    score = similarity_matrix[i, j]

    pairs.append({
        "review_1": df.iloc[i]["feedback"],
        "review_2": df.iloc[j]["feedback"],
        "similarity": score
    })


# Sort highest similarity first
pairs = sorted(
    pairs,
    key=lambda x: x["similarity"],
    reverse=True
)


# Show top 15 pairs
for index, pair in enumerate(pairs[:15], start=1):

    print("\n" + "=" * 80)
    print(f"PAIR {index}")
    print("=" * 80)

    print("Review 1:")
    print(pair["review_1"])

    print("\nReview 2:")
    print(pair["review_2"])

    print(
        "\nCosine similarity:",
        round(pair["similarity"], 4)
    )


print("\nSimilarity check completed!")