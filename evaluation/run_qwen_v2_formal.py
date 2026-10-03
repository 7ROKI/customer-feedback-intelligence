import os
import sys
import time
from pathlib import Path
import pandas as pd

# ============================================================
# 1. Allow importing from src/
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.qwen_v2_formal import classify_with_qwen_v2


# ============================================================
# 2. Configuration
# ============================================================

INPUT_FILE = "data/formal_evaluation_inputs.csv"
GROUND_TRUTH_FILE = "data/formal_ground_truth.csv"

PREDICTION_FILE = "evaluation/qwen_v2_formal_predictions.csv"
METRICS_FILE = "evaluation/qwen_v2_formal_metrics.csv"
RESULT_FILE = "evaluation/qwen_v2_formal_evaluation_results.csv"


CATEGORIES = [
    "Account & Login",
    "Bugs & Technical Issues",
    "Performance",
    "Feature Problem",
    "Feature Request",
    "Privacy / Security",
    "Other / General"
]


# ============================================================
# 3. Load frozen evaluation inputs
# ============================================================

inputs = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print("Formal Qwen V2 evaluation")
print("Total reviews:", len(inputs))


# ============================================================
# 4. Resume support
# ============================================================

if os.path.exists(PREDICTION_FILE):

    existing = pd.read_csv(
        PREDICTION_FILE,
        dtype=str
    )

    completed_ids = set(
        existing["review_id"]
    )

    print(
        f"Existing predictions found: {len(existing)}"
    )

else:

    existing = pd.DataFrame()

    completed_ids = set()

    print(
        "No previous predictions found. Starting fresh."
    )


# ============================================================
# 5. Run Qwen V2
# ============================================================

new_results = []

for index, row in inputs.iterrows():

    review_id = row["review_id"]

    if review_id in completed_ids:
        continue

    feedback = row["feedback"]

    print(
        f"\n[{index + 1}/{len(inputs)}] "
        f"Review ID: {review_id}"
    )

    try:

        (
            prediction,
            input_tokens,
            output_tokens,
            total_tokens
        ) = classify_with_qwen_v2(feedback)

        print(
            "Prediction:",
            prediction
        )

        result = {
            "review_id": review_id,
            "feedback": feedback,
            "predicted_issue_category": prediction,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }

        new_results.append(result)

        # -----------------------------------------
        # Save after every successful API call
        # -----------------------------------------

        new_df = pd.DataFrame(
            new_results
        )

        if len(existing) > 0:

            combined = pd.concat(
                [
                    existing,
                    new_df
                ],
                ignore_index=True
            )

        else:

            combined = new_df

        combined.to_csv(
            PREDICTION_FILE,
            index=False
        )

        time.sleep(0.2)

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        print(
            "Stopping so completed results are not lost."
        )

        break


# ============================================================
# 6. Reload predictions
# ============================================================

predictions = pd.read_csv(
    PREDICTION_FILE,
    dtype={
        "review_id": str,
        "feedback": str,
        "predicted_issue_category": str,
        "input_tokens": float,
        "output_tokens": float,
        "total_tokens": float
    }
)


print(
    "\nPredictions completed:",
    len(predictions),
    "/",
    len(inputs)
)


# ============================================================
# 7. Token usage
# ============================================================

total_input_tokens = (
    predictions["input_tokens"]
    .fillna(0)
    .sum()
)

total_output_tokens = (
    predictions["output_tokens"]
    .fillna(0)
    .sum()
)

total_tokens = (
    predictions["total_tokens"]
    .fillna(0)
    .sum()
)


print(
    "\n=== Token Usage ==="
)

print(
    "Input tokens:",
    int(total_input_tokens)
)

print(
    "Output tokens:",
    int(total_output_tokens)
)

print(
    "Total tokens:",
    int(total_tokens)
)


# ============================================================
# 8. Stop if not all 150 completed
# ============================================================

if len(predictions) != len(inputs):

    print(
        "\nEvaluation not calculated yet "
        "because not all 150 reviews are complete."
    )

    print(
        "Run this script again to resume."
    )

    raise SystemExit


# ============================================================
# 9. Load frozen ground truth
# ============================================================

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
        "issue_category":
        "true_issue_category"
    }
)


results["correct"] = (
    results["predicted_issue_category"]
    ==
    results["true_issue_category"]
)


# ============================================================
# 10. Overall accuracy
# ============================================================

accuracy = (
    results["correct"].mean()
)


# ============================================================
# 11. Per-category metrics
# ============================================================

metrics = []


for category in CATEGORIES:

    tp = (
        (
            results["predicted_issue_category"]
            == category
        )
        &
        (
            results["true_issue_category"]
            == category
        )
    ).sum()

    fp = (
        (
            results["predicted_issue_category"]
            == category
        )
        &
        (
            results["true_issue_category"]
            != category
        )
    ).sum()

    fn = (
        (
            results["predicted_issue_category"]
            != category
        )
        &
        (
            results["true_issue_category"]
            == category
        )
    ).sum()

    support = (
        results["true_issue_category"]
        == category
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
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )


    metrics.append({
        "category": category,
        "support": support,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(
            precision,
            4
        ),
        "recall": round(
            recall,
            4
        ),
        "f1": round(
            f1,
            4
        )
    })


metrics_df = pd.DataFrame(
    metrics
)


macro_precision = (
    metrics_df["precision"]
    .mean()
)

macro_recall = (
    metrics_df["recall"]
    .mean()
)

macro_f1 = (
    metrics_df["f1"]
    .mean()
)


# ============================================================
# 12. Save files
# ============================================================

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

results.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 13. Final output
# ============================================================

print(
    "\n================================="
)

print(
    "QWEN V2 FORMAL EVALUATION"
)

print(
    "================================="
)


print(
    "\n=== Overall ==="
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Macro Precision: "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall: "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)


print(
    "\n=== Per-category metrics ==="
)

print(
    metrics_df[
        [
            "category",
            "support",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n=== Token Usage ==="
)

print(
    "Input tokens:",
    int(total_input_tokens)
)

print(
    "Output tokens:",
    int(total_output_tokens)
)

print(
    "Total tokens:",
    int(total_tokens)
)


print(
    "\nFiles saved:"
)

print(
    PREDICTION_FILE
)

print(
    METRICS_FILE
)

print(
    RESULT_FILE
)