import sys
from pathlib import Path
import pandas as pd

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.baseline import classify_feedback


INPUT_FILE = "data/formal_evaluation_inputs.csv"
GROUND_TRUTH_FILE = "data/formal_ground_truth.csv"

OUTPUT_FILE = "evaluation/formal_baseline_predictions.csv"
METRICS_FILE = "evaluation/formal_baseline_metrics.csv"


# Map the old toy-baseline labels
# to the frozen formal taxonomy
CATEGORY_MAP = {
    "Payment": "Other / General",
    "Account": "Account & Login",
    "Performance": "Performance",
    "Technical": "Bugs & Technical Issues",
    "Other": "Other / General"
}


FORMAL_CATEGORIES = [
    "Account & Login",
    "Bugs & Technical Issues",
    "Performance",
    "Feature Problem",
    "Feature Request",
    "Privacy / Security",
    "Other / General"
]


# ----------------------------
# 1. Load frozen evaluation data
# ----------------------------

inputs = pd.read_csv(INPUT_FILE, dtype=str)
ground_truth = pd.read_csv(GROUND_TRUTH_FILE, dtype=str)


# ----------------------------
# 2. Run deterministic baseline
# ----------------------------

predictions = []

for _, row in inputs.iterrows():

    old_prediction = classify_feedback(row["feedback"])

    formal_prediction = CATEGORY_MAP[old_prediction]

    predictions.append({
        "review_id": row["review_id"],
        "feedback": row["feedback"],
        "baseline_raw_category": old_prediction,
        "predicted_issue_category": formal_prediction
    })


pred_df = pd.DataFrame(predictions)


# ----------------------------
# 3. Join predictions with truth
# ----------------------------

results = pred_df.merge(
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
    == results["true_issue_category"]
)


# ----------------------------
# 4. Overall accuracy
# ----------------------------

accuracy = results["correct"].mean()


# ----------------------------
# 5. Per-category metrics
# ----------------------------

metrics = []

for category in FORMAL_CATEGORIES:

    true_positive = (
        (results["predicted_issue_category"] == category)
        &
        (results["true_issue_category"] == category)
    ).sum()

    false_positive = (
        (results["predicted_issue_category"] == category)
        &
        (results["true_issue_category"] != category)
    ).sum()

    false_negative = (
        (results["predicted_issue_category"] != category)
        &
        (results["true_issue_category"] == category)
    ).sum()

    support = (
        results["true_issue_category"] == category
    ).sum()

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
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
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    })


metrics_df = pd.DataFrame(metrics)


# Macro averages
macro_precision = metrics_df["precision"].mean()
macro_recall = metrics_df["recall"].mean()
macro_f1 = metrics_df["f1"].mean()


# ----------------------------
# 6. Save results
# ----------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ----------------------------
# 7. Print summary
# ----------------------------

print("Formal baseline evaluation completed!")

print("\n=== Dataset ===")
print("Total reviews:", len(results))

print("\n=== Overall ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\n=== Per-category metrics ===")
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

print("\nPredictions saved to:")
print(OUTPUT_FILE)

print("\nMetrics saved to:")
print(METRICS_FILE)