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

PREDICTION_FILE = "evaluation/qwen_severity_predictions.csv"
RESULT_FILE = "evaluation/qwen_severity_evaluation_results.csv"
METRICS_FILE = "evaluation/qwen_severity_metrics.csv"

VALID_LABELS = ["High", "Medium", "Low"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


# ============================================================
# 2. Severity classifier
# ============================================================

def classify_severity(feedback):

    prompt = f"""
Classify the severity of the following customer feedback.

Choose exactly ONE label:

High
- The user cannot complete a core task.
- Account access, login, signup, verification, or security is seriously affected.
- There is severe loss of important functionality or risk of data loss.
- There is a serious privacy, fraud, scam, or security concern.

Medium
- An important feature is broken or degraded.
- The problem significantly affects the experience.
- The app is still generally usable.

Low
- Minor inconvenience or experience issue.
- Feature request or preference.
- General praise or dissatisfaction.
- No concrete serious product issue.

Important rules:
- Judge severity from the feedback text, not from the star rating.
- Do not treat negative emotion alone as High severity.
- If the user cannot access their account or complete a core communication task, prefer High.
- If an existing feature is broken but the app remains generally usable, prefer Medium.
- If the feedback is vague, general, positive, or merely a request for change, prefer Low.

Customer feedback:
{feedback}

Return only one exact label:
High
Medium
Low
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
# 3. Load frozen inputs
# ============================================================

inputs = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print("Formal Qwen Severity Evaluation")
print("Total reviews:", len(inputs))
print("Model:", MODEL)


# ============================================================
# 4. Resume support
# ============================================================

if os.path.exists(PREDICTION_FILE):

    existing = pd.read_csv(
        PREDICTION_FILE,
        dtype=str
    )

    completed_ids = set(existing["review_id"])

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

        (
            prediction,
            input_tokens,
            output_tokens,
            total_tokens
        ) = classify_severity(feedback)

        print("Prediction:", prediction)

        result = {
            "review_id": review_id,
            "feedback": feedback,
            "predicted_severity": prediction,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }

        new_results.append(result)

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

        time.sleep(0.2)

    except Exception as e:

        print("ERROR:", e)
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
        "predicted_severity": str,
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
# 8. Stop if incomplete
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
# 9. Normalize output
# ============================================================

predictions["predicted_severity"] = (
    predictions["predicted_severity"]
    .str.strip()
    .replace({
        "1. High": "High",
        "2. Medium": "Medium",
        "3. Low": "Low"
    })
)

invalid_predictions = predictions[
    ~predictions["predicted_severity"].isin(
        VALID_LABELS
    )
]

print(
    "\nInvalid / abstained predictions:",
    len(invalid_predictions)
)


# ============================================================
# 10. Load frozen ground truth
# ============================================================

ground_truth = pd.read_csv(
    GROUND_TRUTH_FILE,
    dtype=str
)

results = predictions.merge(
    ground_truth[
        [
            "review_id",
            "severity"
        ]
    ],
    on="review_id",
    how="left"
)

results = results.rename(
    columns={
        "severity": "true_severity"
    }
)


# ============================================================
# 11. Overall 3-class accuracy
# ============================================================

results["correct"] = (
    results["predicted_severity"]
    ==
    results["true_severity"]
)

accuracy = results["correct"].mean()


# ============================================================
# 12. High-priority binary evaluation
# ============================================================

# High is positive class.
# Medium + Low are negative class.
# Invalid / abstained output:
# - if true High: counts as false negative
# - if true non-High: not counted as false positive

true_high = (
    results["true_severity"] == "High"
)

pred_high = (
    results["predicted_severity"] == "High"
)

valid_prediction = (
    results["predicted_severity"].isin(
        VALID_LABELS
    )
)

tp = (
    true_high
    &
    pred_high
).sum()

fp = (
    (~true_high)
    &
    pred_high
).sum()

fn = (
    true_high
    &
    (~pred_high)
).sum()

tn = (
    (~true_high)
    &
    valid_prediction
    &
    (~pred_high)
).sum()

abstentions = (
    ~valid_prediction
).sum()

high_support = true_high.sum()

predicted_high_count = pred_high.sum()


high_precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

high_recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

high_f1 = (
    2
    * high_precision
    * high_recall
    / (high_precision + high_recall)
    if (high_precision + high_recall) > 0
    else 0
)


# ============================================================
# 13. Per-severity metrics
# ============================================================

severity_metrics = []

for label in VALID_LABELS:

    label_tp = (
        (results["predicted_severity"] == label)
        &
        (results["true_severity"] == label)
    ).sum()

    label_fp = (
        (results["predicted_severity"] == label)
        &
        (results["true_severity"] != label)
    ).sum()

    label_fn = (
        (results["predicted_severity"] != label)
        &
        (results["true_severity"] == label)
    ).sum()

    support = (
        results["true_severity"] == label
    ).sum()

    precision = (
        label_tp / (label_tp + label_fp)
        if (label_tp + label_fp) > 0
        else 0
    )

    recall = (
        label_tp / (label_tp + label_fn)
        if (label_tp + label_fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    severity_metrics.append({
        "severity": label,
        "support": support,
        "true_positive": label_tp,
        "false_positive": label_fp,
        "false_negative": label_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    })


severity_metrics_df = pd.DataFrame(
    severity_metrics
)


# ============================================================
# 14. Save files
# ============================================================

results.to_csv(
    RESULT_FILE,
    index=False
)

severity_metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# 15. Print final results
# ============================================================

print(
    "\n================================="
)

print(
    "QWEN SEVERITY FORMAL EVALUATION"
)

print(
    "================================="
)


print(
    "\n=== Overall 3-Class Severity ==="
)

print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    "\n=== High-Priority Evaluation ==="
)

print(
    "True High cases:",
    int(high_support)
)

print(
    "Predicted High cases:",
    int(predicted_high_count)
)

print(
    "True Positives:",
    int(tp)
)

print(
    "False Positives / False Alarms:",
    int(fp)
)

print(
    "False Negatives / Missed High cases:",
    int(fn)
)

print(
    "True Negatives:",
    int(tn)
)

print(
    "Abstentions / Invalid outputs:",
    int(abstentions)
)

print(
    f"High Precision: {high_precision:.4f}"
)

print(
    f"High Recall: {high_recall:.4f}"
)

print(
    f"High F1: {high_f1:.4f}"
)


print(
    "\n=== Per-Severity Metrics ==="
)

print(
    severity_metrics_df[
        [
            "severity",
            "support",
            "precision",
            "recall",
            "f1"
        ]
    ].to_string(index=False)
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

print(PREDICTION_FILE)
print(RESULT_FILE)
print(METRICS_FILE)