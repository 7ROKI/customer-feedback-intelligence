import pandas as pd

FILE = "evaluation/qwen_severity_evaluation_results.csv"

df = pd.read_csv(FILE, dtype=str)

print("=== Severity Confusion Matrix ===")

confusion = pd.crosstab(
    df["true_severity"],
    df["predicted_severity"]
)

print(confusion.to_string())


print("\n=== Medium Severity Errors ===")

medium_errors = df[
    (df["true_severity"] == "Medium")
    &
    (df["predicted_severity"] != "Medium")
]

for _, row in medium_errors.iterrows():

    print("\nReview ID:", row["review_id"])
    print("Feedback:", row["feedback"])
    print("True:", row["true_severity"])
    print("Predicted:", row["predicted_severity"])
    print("-" * 70)


print("\nMedium errors:", len(medium_errors))