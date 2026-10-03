import pandas as pd


# Load cleaned dataset
df = pd.read_csv("data/cleaned_reviews.csv")


# Create a simple rating group
def rating_group(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


df["rating_group"] = df["rating"].apply(rating_group)


# Target sample size
TARGET_SIZE = 150


# Stratified sample by app and rating group
sampled_parts = []

groups = df.groupby(["app_id", "rating_group"])

for _, group in groups:
    n = max(1, round(len(group) / len(df) * TARGET_SIZE))

    n = min(n, len(group))

    sampled = group.sample(
        n=n,
        random_state=42
    )

    sampled_parts.append(sampled)


evaluation_df = pd.concat(sampled_parts)


# If rounding gives slightly more than 150 rows, trim
if len(evaluation_df) > TARGET_SIZE:
    evaluation_df = evaluation_df.sample(
        n=TARGET_SIZE,
        random_state=42
    )


# If rounding gives fewer than 150 rows, fill from remaining rows
if len(evaluation_df) < TARGET_SIZE:
    remaining = df.drop(evaluation_df.index)

    extra_needed = TARGET_SIZE - len(evaluation_df)

    extra = remaining.sample(
        n=extra_needed,
        random_state=42
    )

    evaluation_df = pd.concat(
        [evaluation_df, extra]
    )


# Shuffle final set
evaluation_df = evaluation_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Add manual annotation columns
evaluation_df["issue_category"] = ""
evaluation_df["specific_issue"] = ""
evaluation_df["sentiment"] = ""
evaluation_df["severity"] = ""


# Keep useful columns
evaluation_df = evaluation_df[
    [
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
]


# Save
evaluation_df.to_csv(
    "data/formal_evaluation_set.csv",
    index=False,
    encoding="utf-8"
)


print("Formal evaluation set created!")
print("Total rows:", len(evaluation_df))
print()
print("Rating distribution:")
print(evaluation_df["rating"].value_counts().sort_index())
print()
print("Reviews by app:")
print(evaluation_df["app_id"].value_counts())
print()
print("Saved to data/formal_evaluation_set.csv")