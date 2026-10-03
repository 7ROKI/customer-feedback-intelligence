import pandas as pd

PREDICTION_FILE = "evaluation/qwen_v2_formal_predictions.csv"
GROUND_TRUTH_FILE = "data/formal_ground_truth.csv"
OUTPUT_RESULT_FILE = "evaluation/qwen_v2_formal_evaluation_results_normalized.csv"
OUTPUT_METRICS_FILE = "evaluation/qwen_v2_formal_metrics_normalized.csv"

CATEGORIES = [
    "Account & Login",
    "Bugs & Technical Issues",
    "Performance",
    "Feature Problem",
    "Feature Request",
    "Privacy / Security",
    "Other / General"
]

# Load predictions
predictions = pd.read_csv(PREDICTION_FILE, dtype=str)

# Normalize only known formatting variation
predictions["predicted_issue_category"] = (
    predictions["predicted_issue_category"]
    .replace({
        "7. Other / General": "Other / General"
    })
)

print("=== Unique normalized predictions ===")
print(sorted(predictions["predicted_issue_category"].dropna().unique()))

# Load frozen ground truth
ground_truth = pd.read_csv(
    GROUND_TRUTH_FILE,
    dtype=str
)

results = predictions.merge(
    ground_truth[
        [
            "review_id",
            "issue_category"
        ]
    ],
    on="review_id",
    how="left"
)

results = results.rename(
    columns={
        "issue_category": "true_issue_category"
    }
)

results["correct"] = (
    results["predicted_issue_category"]
    ==
    results["true_issue_category"]
)

accuracy = results["correct"].mean()

metrics = []

for category in CATEGORIES:

    tp = (
        (results["predicted_issue_category"] == category)
        &
        (results["true_issue_category"] == category)
    ).sum()

    fp = (
        (results["predicted_issue_category"] == category)
        &
        (results["true_issue_category"] != category)
    ).sum()

    fn = (
        (results["predicted_issue_category"] != category)
        &
        (results["true_issue_category"] == category)
    ).sum()

    support = (
        results["true_issue_category"] == category
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics.append({
        "category": category,
        "support": support,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    })

metrics_df = pd.DataFrame(metrics)

macro_precision = metrics_df["precision"].mean()
macro_recall = metrics_df["recall"].mean()
macro_f1 = metrics_df["f1"].mean()

results.to_csv(
    OUTPUT_RESULT_FILE,
    index=False
)

metrics_df.to_csv(
    OUTPUT_METRICS_FILE,
    index=False
)

print("\n=== Normalized Overall ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\n=== Normalized Per-category Metrics ===")
print(
    metrics_df[
        [
            "category",
            "support",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_RESULT_FILE)
print(OUTPUT_METRICS_FILE)