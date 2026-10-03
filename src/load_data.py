from datasets import load_dataset
import pandas as pd


print("Loading dataset...")

dataset = load_dataset(
    "UniqueData/messengers-reviews-google-play",
    split="train"
)

df = dataset.to_pandas()

print("Dataset loaded successfully!")
print("Total rows:", len(df))
print()

# Basic information
print("Columns:")
print(df.columns.tolist())
print()

print("Unique language values:")
print(df["userLang"].value_counts(dropna=False).head(20))
print()
# English reviews
df["userLang"] = df["userLang"].astype(str).str.strip().str.upper()

english_df = df[df["userLang"] == "EN"].copy()

print("English reviews:", len(english_df))
print()

# Missing content
missing_content = english_df["content"].isna().sum()

print("Missing review text:", missing_content)
print()

# Duplicate reviews
duplicate_count = english_df["content"].duplicated().sum()

print("Duplicate review texts:", duplicate_count)
print()

# Review length
english_df["review_length"] = english_df["content"].astype(str).str.len()

print("Average review length:", round(english_df["review_length"].mean(), 2))
print("Shortest review length:", english_df["review_length"].min())
print("Longest review length:", english_df["review_length"].max())
print()

# App distribution
print("Reviews by app:")
print(english_df["app_id"].value_counts())