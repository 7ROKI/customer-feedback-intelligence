import pandas as pd


df = pd.read_csv("data/formal_evaluation_set.csv")


# 找还没有标注 issue_category 的行
unlabeled = df[
    df["issue_category"].isna()
].head(10)


print("Next 10 unlabeled reviews:")
print()


for _, row in unlabeled.iterrows():
    print("Review ID:", row["review_id"])
    print("Rating:", row["rating"])
    print("App:", row["app_id"])
    print("Feedback:", row["feedback"])
    print("-" * 70)


print()
print("Remaining unlabeled:", df["issue_category"].isna().sum())