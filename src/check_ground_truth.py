import pandas as pd

FILE = "data/formal_ground_truth.csv"

df = pd.read_csv(FILE, dtype=str)

print("Total rows:", len(df))

print("\n=== Issue Category Distribution ===")
print(df["issue_category"].value_counts(dropna=False))

print("\n=== Sentiment Distribution ===")
print(df["sentiment"].value_counts(dropna=False))

print("\n=== Severity Distribution ===")
print(df["severity"].value_counts(dropna=False))

print("\n=== Missing Values ===")
print(df.isna().sum())

print("\n=== Duplicate Review IDs ===")
print(df["review_id"].duplicated().sum())

print("\n=== Unique Issue Categories ===")
print(sorted(df["issue_category"].dropna().unique()))

print("\n=== Unique Sentiments ===")
print(sorted(df["sentiment"].dropna().unique()))

print("\n=== Unique Severities ===")
print(sorted(df["severity"].dropna().unique()))