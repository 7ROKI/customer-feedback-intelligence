import pandas as pd

df = pd.read_csv("data/clustered_specific_issues.csv")

cluster_ids = [0, 31, 151, 39, 42, 17, 118, 152]

for cid in cluster_ids:
    part = df[df["cluster_id"] == cid]

    if len(part) == 0:
        continue

    print("\n" + "=" * 80)

    print(
        "CLUSTER:",
        cid,
        "| CATEGORY:",
        part["issue_category"].iloc[0],
        "| SIZE:",
        len(part)
    )

    print("=" * 80)

    for x in part["specific_issue"].tolist():
        print("-", x)