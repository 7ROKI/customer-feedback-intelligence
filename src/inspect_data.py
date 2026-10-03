import pandas as pd


df = pd.read_csv("data/cleaned_reviews.csv")


print("Total cleaned reviews:", len(df))
print()


# 1. Rating distribution
print("Rating distribution:")
print(df["rating"].value_counts().sort_index())
print()


# 2. Reviews by app
print("Reviews by app:")
print(df["app_id"].value_counts())
print()


# 3. Sentiment-style grouping by rating
negative = df[df["rating"] <= 2]
neutral = df[df["rating"] == 3]
positive = df[df["rating"] >= 4]

print("Negative reviews (1-2 stars):", len(negative))
print("Neutral reviews (3 stars):", len(neutral))
print("Positive reviews (4-5 stars):", len(positive))
print()


# 4. Random sample
print("Random sample of 20 reviews:")
sample = df.sample(n=20, random_state=42)

for _, row in sample.iterrows():
    print("App:", row["app_id"])
    print("Rating:", row["rating"])
    print("Feedback:", row["feedback"])
    print("-" * 60)