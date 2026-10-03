import pandas as pd

RESULT_FILE = "evaluation/qwen_v1_formal_evaluation_results.csv"

df = pd.read_csv(RESULT_FILE, dtype=str)

# Convert correct column to boolean safely
df["correct"] = df["correct"].astype(str).str.lower() == "true"

print("Total rows:", len(df))
print("Correct:", df["correct"].sum())
print("Incorrect:", (~df["correct"]).sum())


# ============================================================
# 1. Confusion table
# ============================================================

confusion = pd.crosstab(
    df["true_issue_category"],
    df["predicted_issue_category"]
)

print("\n=== Confusion Matrix ===")
print(confusion.to_string())


# ============================================================
# 2. Error counts by true category
# ============================================================

errors = df[~df["correct"]].copy()

error_summary = (
    errors.groupby(
        [
            "true_issue_category",
            "predicted_issue_category"
        ]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        ["true_issue_category", "count"],
        ascending=[True, False]
    )
)

print("\n=== Error Summary ===")
print(error_summary.to_string(index=False))


# ============================================================
# 3. Show detailed mistakes
# ============================================================

print("\n=== Detailed Errors ===")

for true_category in sorted(
    errors["true_issue_category"].dropna().unique()
):

    subset = errors[
        errors["true_issue_category"] == true_category
    ]

    print("\n")
    print("=" * 80)
    print("TRUE CATEGORY:", true_category)
    print("=" * 80)

    for _, row in subset.iterrows():

        print("\nReview ID:", row["review_id"])
        print("Feedback:", row["feedback"])
        print("True:", row["true_issue_category"])
        print("Predicted:", row["predicted_issue_category"])
        print("-" * 80)


# ============================================================
# 4. Save error files
# ============================================================

errors.to_csv(
    "evaluation/qwen_v1_errors.csv",
    index=False
)

error_summary.to_csv(
    "evaluation/qwen_v1_error_summary.csv",
    index=False
)

confusion.to_csv(
    "evaluation/qwen_v1_confusion_matrix.csv"
)

print("\nFiles saved:")
print("evaluation/qwen_v1_errors.csv")
print("evaluation/qwen_v1_error_summary.csv")
print("evaluation/qwen_v1_confusion_matrix.csv")