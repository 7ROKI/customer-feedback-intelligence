import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

INPUT_FILE = "data/formal_evaluation_inputs.csv"
GROUND_TRUTH_FILE = "data/formal_ground_truth.csv"

PREDICTION_FILE = "evaluation/qwen_v1_formal_predictions.csv"
METRICS_FILE = "evaluation/qwen_v1_formal_metrics.csv"


CATEGORIES = [
    "Account & Login",
    "Bugs & Technical Issues",
    "Performance",
    "Feature Problem",
    "Feature Request",
    "Privacy / Security",
    "Other / General"
]


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


# ============================================================
# 2. V1 prompt
# ============================================================

def classify_with_qwen(feedback):

    prompt = f"""
Classify the following customer feedback into exactly ONE category.

Categories:
- Account & Login
- Bugs & Technical Issues
- Performance
- Feature Problem
- Feature Request
- Privacy / Security
- Other / General

Customer feedback:
{feedback}

Return only the category name.
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    prediction = response.choices[0].message.content.strip()

    usage = response.usage

    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    return (
        prediction,
        input_tokens,
        output_tokens,
        total_tokens
    )


# ============================================================
# 3. Load frozen model inputs
# ============================================================

inputs = pd.read_csv(INPUT_FILE, dtype=str)

print("Formal Qwen V1 evaluation")
print("Total reviews:", len(inputs))
print("Model:", MODEL)


# ============================================================
# 4. Resume support
# ============================================================

if os.path.exists(PREDICTION_FILE):

    existing = pd.read_csv(PREDICTION_FILE, dtype=str)

    completed_ids = set(existing["review_id"])

    print(
        f"Existing predictions found: {len(existing)}"
    )

else:

    existing = pd.DataFrame()

    completed_ids = set()

    print("No previous predictions found. Starting fresh.")


# ============================================================
# 5. Run model
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

        prediction, input_tokens, output_tokens, total_tokens = (
            classify_with_qwen(feedback)
        )

        print("Prediction:", prediction)

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

        new_df = pd.DataFrame(new_results)

        if len(existing) > 0:
            combined = pd.concat(
                [existing, new_df],
                ignore_index=True
            )
        else:
            combined = new_df

        combined.to_csv(
            PREDICTION_FILE,
            index=False
        )

        # Small delay for stability
        time.sleep(0.2)

    except Exception as e:

        print("ERROR:", e)
        print("Stopping so completed results are not lost.")
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

total_input_tokens = predictions["input_tokens"].fillna(0).sum()
total_output_tokens = predictions["output_tokens"].fillna(0).sum()
total_tokens = predictions["total_tokens"].fillna(0).sum()

print("\n=== Token Usage ===")

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
# 8. Only evaluate if all 150 predictions completed
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
# 9. Load frozen ground truth AFTER predictions
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
        "issue_category": "true_issue_category"
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

accuracy = results["correct"].mean()


# ============================================================
# 11. Per-category metrics
# ============================================================

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


# ============================================================
# 12. Save final metrics
# ============================================================

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


results.to_csv(
    "evaluation/qwen_v1_formal_evaluation_results.csv",
    index=False
)


# ============================================================
# 13. Print final results
# ============================================================

print("\n=================================")
print("QWEN V1 FORMAL EVALUATION")
print("=================================")

print("\n=== Overall ===")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Macro Precision: {macro_precision:.4f}"
)

print(
    f"Macro Recall: {macro_recall:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


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


print("\n=== Token Usage ===")

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


print("\nFiles saved:")

print(
    PREDICTION_FILE
)

print(
    METRICS_FILE
)

print(
    "evaluation/qwen_v1_formal_evaluation_results.csv"
)