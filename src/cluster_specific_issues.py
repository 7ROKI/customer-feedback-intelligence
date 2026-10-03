import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# ============================================================
# 1. Configuration
# ============================================================

INPUT_CSV = "data/specific_issues_for_clustering.csv"
INPUT_NPY = "data/specific_issue_embeddings.npy"

OUTPUT_FILE = "data/clustered_specific_issues.csv"

# cosine similarity ~ 0.75
# cosine distance = 1 - similarity
DISTANCE_THRESHOLD = 0.28


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)
embeddings = np.load(INPUT_NPY)
# ============================================================
# Remove clearly non-specific "general" descriptions
# that were inconsistently marked as actionable by the LLM
# ============================================================

general_mask = (
    df["specific_issue"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.startswith("general ")
)

removed_count = general_mask.sum()

print(
    "Removed non-specific general descriptions:",
    removed_count
)

df = df[~general_mask].copy()

embeddings = embeddings[~general_mask.to_numpy()]

df = df.reset_index(drop=True)

print("Rows:", len(df))
print("Embedding shape:", embeddings.shape)

if len(df) != len(embeddings):
    raise ValueError(
        "CSV rows and embedding rows do not match."
    )


# ============================================================
# 3. Normalize embeddings
# ============================================================

normalized_embeddings = normalize(
    embeddings,
    norm="l2"
)


# ============================================================
# 4. Cluster within each issue category
# ============================================================

all_results = []

global_cluster_id = 0


for category in sorted(
    df["issue_category"].dropna().unique()
):

    category_mask = (
        df["issue_category"] == category
    )

    category_df = (
        df[category_mask]
        .copy()
        .reset_index()
    )

    category_indices = (
        np.where(category_mask)[0]
    )

    category_embeddings = (
        normalized_embeddings[
            category_indices
        ]
    )

    print("\n" + "=" * 80)
    print("CATEGORY:", category)
    print("Reviews:", len(category_df))
    print("=" * 80)


    # --------------------------------------------------------
    # If category contains only one review,
    # assign one cluster directly
    # --------------------------------------------------------

    if len(category_df) == 1:

        local_labels = np.array([0])

    else:

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=DISTANCE_THRESHOLD
        )

        local_labels = clustering.fit_predict(
            category_embeddings
        )


    # --------------------------------------------------------
    # Map local cluster labels to global unique IDs
    # --------------------------------------------------------

    unique_local_labels = sorted(
        np.unique(local_labels)
    )

    local_to_global = {}

    for local_label in unique_local_labels:

        local_to_global[local_label] = (
            global_cluster_id
        )

        global_cluster_id += 1


    category_df["local_cluster_id"] = (
        local_labels
    )

    category_df["cluster_id"] = (
        category_df["local_cluster_id"]
        .map(local_to_global)
    )


    print(
        "Clusters:",
        len(unique_local_labels)
    )


    # --------------------------------------------------------
    # Print cluster contents
    # --------------------------------------------------------

    for local_label in unique_local_labels:

        cluster_rows = category_df[
            category_df["local_cluster_id"]
            == local_label
        ]

        print("\nCluster", local_label)
        print(
            "Size:",
            len(cluster_rows)
        )

        for issue in (
            cluster_rows["specific_issue"]
            .head(8)
        ):

            print(
                " -",
                issue
            )

        if len(cluster_rows) > 8:

            print(
                " ...",
                len(cluster_rows) - 8,
                "more"
            )


    all_results.append(
        category_df
    )


# ============================================================
# 5. Combine all categories
# ============================================================

result_df = pd.concat(
    all_results,
    ignore_index=True
)


# Remove temporary original-index column
if "index" in result_df.columns:
    result_df = result_df.drop(
        columns=["index"]
    )


# ============================================================
# 6. Save
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 7. Summary
# ============================================================

cluster_summary = (
    result_df
    .groupby(
        [
            "issue_category",
            "cluster_id"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


print("\n" + "=" * 80)
print("CLUSTERING SUMMARY")
print("=" * 80)

print(
    "Total actionable reviews:",
    len(result_df)
)

print(
    "Total clusters:",
    result_df["cluster_id"].nunique()
)

print(
    "\nClusters by category:"
)

print(
    cluster_summary
    .groupby("issue_category")
    .size()
    .sort_values(
        ascending=False
    )
)


print(
    "\nLargest clusters:"
)

print(
    cluster_summary
    .sort_values(
        "count",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)