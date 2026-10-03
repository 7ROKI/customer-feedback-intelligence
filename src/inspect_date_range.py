import pandas as pd

FILE = "data/clustered_specific_issues.csv"

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("Total rows:", len(df))

print("\n=== Date Range ===")
print("Earliest:", df["date"].min())
print("Latest:", df["date"].max())

print("\n=== Missing Dates ===")
print(df["date"].isna().sum())

print("\n=== Reviews by Month ===")

monthly = (
    df
    .dropna(subset=["date"])
    .assign(
        month=lambda x:
        x["date"].dt.to_period("M")
    )
    .groupby("month")
    .size()
)

print(monthly)