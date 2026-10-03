import pandas as pd

INPUT_FILE = "data/formal_evaluation_set.csv"
INPUTS_OUTPUT = "data/formal_evaluation_inputs.csv"
GROUND_TRUTH_OUTPUT = "data/formal_ground_truth.csv"

# Read the fully annotated evaluation set
df = pd.read_csv(INPUT_FILE, dtype=str)

# Basic completeness check
required_columns = [
    "review_id",
    "feedback",
    "rating",
    "date",
    "app_id",
    "issue_category",
    "specific_issue",
    "sentiment",
    "severity"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Check that all 150 rows have complete labels
label_columns = [
    "issue_category",
    "specific_issue",
    "sentiment",
    "severity"
]

missing_labels = df[label_columns].isna().sum()

print("Rows in formal evaluation set:", len(df))
print("\nMissing labels:")
print(missing_labels)

if len(df) != 150:
    raise ValueError(f"Expected 150 rows, but found {len(df)}")

if missing_labels.sum() > 0:
    raise ValueError("Some ground-truth labels are still missing.")

# Create model input file: NO ground-truth labels
inputs_df = df[
    [
        "review_id",
        "feedback",
        "rating",
        "date",
        "app_id"
    ]
].copy()

# Create separate frozen ground-truth file
ground_truth_df = df[
    [
        "review_id",
        "issue_category",
        "specific_issue",
        "sentiment",
        "severity"
    ]
].copy()

inputs_df.to_csv(INPUTS_OUTPUT, index=False)
ground_truth_df.to_csv(GROUND_TRUTH_OUTPUT, index=False)

print("\nGround truth frozen successfully!")
print(f"Input file saved to: {INPUTS_OUTPUT}")
print(f"Ground truth saved to: {GROUND_TRUTH_OUTPUT}")
print(f"Input rows: {len(inputs_df)}")
print(f"Ground-truth rows: {len(ground_truth_df)}")