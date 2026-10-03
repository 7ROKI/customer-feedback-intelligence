import pandas as pd

FILE = "data/all_reviews_structured.csv"

df = pd.read_csv(FILE)

print("Total reviews:", len(df))

print("\n=== Has Specific Issue ===")
print(df["has_specific_issue"].value_counts(dropna=False))

print("\n=== Issue Category Distribution ===")
print(df["issue_category"].value_counts(dropna=False))

print("\n=== Sentiment Distribution ===")
print(df["sentiment"].value_counts(dropna=False))

print("\n=== Severity Distribution ===")
print(df["severity"].value_counts(dropna=False))

print("\n=== Missing Values ===")
print(
    df[
        [
            "issue_category",
            "specific_issue",
            "sentiment",
            "severity",
            "has_specific_issue"
        ]
    ].isna().sum()
)

specific = df[
    df["has_specific_issue"].astype(str).str.lower() == "true"
]

print("\n=== Reviews For Clustering ===")
print("Specific-issue reviews:", len(specific))
print(
    "Excluded general/non-specific reviews:",
    len(df) - len(specific)
)

print("\n=== Sample Specific Issues ===")

for issue in specific["specific_issue"].head(20):
    print("-", issue)