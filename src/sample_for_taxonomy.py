import pandas as pd


df = pd.read_csv("data/cleaned_reviews.csv")

sample = df.sample(n=30, random_state=42).copy()

sample["manual_category"] = ""

sample.to_csv(
    "data/taxonomy_sample.csv",
    index=False,
    encoding="utf-8"
)

print("Sample created successfully!")
print("Rows:", len(sample))
print("Saved to data/taxonomy_sample.csv")