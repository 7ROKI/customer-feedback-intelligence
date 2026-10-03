from datasets import load_dataset
import pandas as pd


print("Loading dataset...")

dataset = load_dataset(
    "UniqueData/messengers-reviews-google-play",
    split="train"
)

df = dataset.to_pandas()

print("Original rows:", len(df))


# 1. Clean language field
df["userLang"] = df["userLang"].astype(str).str.strip().str.upper()


# 2. Keep English reviews only
df = df[df["userLang"] == "EN"].copy()

print("After English filter:", len(df))


# 3. Remove missing review text
df = df.dropna(subset=["content"])

print("After removing missing content:", len(df))


# 4. Remove duplicate review text
df = df.drop_duplicates(subset=["content"])

print("After removing duplicates:", len(df))


# 5. Clean review text
df["content"] = df["content"].astype(str).str.strip()


# 6. Remove very short reviews
df["review_length"] = df["content"].str.len()

df = df[df["review_length"] >= 10].copy()

print("After removing short reviews:", len(df))


# 7. Keep only useful columns
cleaned_df = df[
    [
        "reviewId",
        "content",
        "score",
        "at",
        "app_id"
    ]
].copy()


# 8. Rename columns to simpler names
cleaned_df = cleaned_df.rename(
    columns={
        "reviewId": "review_id",
        "content": "feedback",
        "score": "rating",
        "at": "date"
    }
)


# 9. Save cleaned dataset
cleaned_df.to_csv(
    "data/cleaned_reviews.csv",
    index=False,
    encoding="utf-8"
)


print()
print("Cleaning completed!")
print("Final rows:", len(cleaned_df))
print("Saved to data/cleaned_reviews.csv")