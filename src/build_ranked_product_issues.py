"""
Build ranked product issues from semantic clusters.

Input:
    data/clustered_specific_issues.csv

Output:
    data/ranked_product_issues.csv

The module calculates:
- issue counts
- severity mix
- deterministic priority score
- three-day trend labels
- representative issue names
- customer evidence

No LLM or external API call is made in this module.
"""
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "data/clustered_specific_issues.csv"

OUTPUT_FILE = "data/ranked_product_issues.csv"


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("Input rows:", len(df))
print(
    "Clusters:",
    df["cluster_id"].nunique()
)


# ============================================================
# 3. Severity weights
# ============================================================

severity_weight = {
    "High": 3,
    "Medium": 2,
    "Low": 1
}

df["severity_weight"] = (
    df["severity"]
    .map(severity_weight)
    .fillna(1)
)


# ============================================================
# 4. Create three equal 3-day trend windows
# ============================================================

start_date = df["date"].min().normalize()

window_1_end = (
    start_date
    + pd.Timedelta(days=2)
)

window_2_start = (
    start_date
    + pd.Timedelta(days=3)
)

window_2_end = (
    start_date
    + pd.Timedelta(days=5)
)

window_3_start = (
    start_date
    + pd.Timedelta(days=6)
)

window_3_end = (
    start_date
    + pd.Timedelta(days=8)
    + pd.Timedelta(hours=23, minutes=59, seconds=59)
)


print("\nTrend windows:")

print(
    "Window 1:",
    start_date.date(),
    "to",
    window_1_end.date()
)

print(
    "Window 2:",
    window_2_start.date(),
    "to",
    window_2_end.date()
)

print(
    "Window 3:",
    window_3_start.date(),
    "to",
    window_3_end.date()
)


# ============================================================
# 5. Representative issue name
# ============================================================

def representative_issue(group):

    counts = (
        group["specific_issue"]
        .value_counts()
    )

    max_count = counts.max()

    candidates = (
        counts[
            counts == max_count
        ]
        .index
        .tolist()
    )

    # If tied, choose shortest clear phrase
    candidates = sorted(
        candidates,
        key=lambda x: (
            len(str(x)),
            str(x)
        )
    )

    return candidates[0]


# ============================================================
# 6. Trend calculation
# ============================================================

def calculate_trend(
    previous_count,
    recent_count
):

    if previous_count == 0 and recent_count > 0:
        return "New"

    if recent_count > previous_count:
        return "Rising"

    if recent_count < previous_count:
        return "Falling"

    return "Stable"


# ============================================================
# 7. Evidence selection
# ============================================================

def select_evidence(group, n=3):

    severity_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    temp = group.copy()

    temp["evidence_priority"] = (
        temp["severity"]
        .map(severity_order)
        .fillna(1)
    )

    temp = (
        temp
        .sort_values(
            [
                "evidence_priority",
                "date"
            ],
            ascending=[
                False,
                False
            ]
        )
        .drop_duplicates(
            subset=["feedback"]
        )
    )

    examples = (
        temp["feedback"]
        .astype(str)
        .head(n)
        .tolist()
    )

    return " || ".join(examples)


# ============================================================
# 8. Build cluster-level summary
# ============================================================

results = []


for cluster_id, group in df.groupby(
    "cluster_id"
):

    category = (
        group["issue_category"]
        .iloc[0]
    )

    issue_name = (
        representative_issue(group)
    )

    total_count = len(group)


    high_count = (
        group["severity"]
        .eq("High")
        .sum()
    )

    medium_count = (
        group["severity"]
        .eq("Medium")
        .sum()
    )

    low_count = (
        group["severity"]
        .eq("Low")
        .sum()
    )


    priority_score = (
        high_count * 3
        + medium_count * 2
        + low_count
    )


    # ----------------------------------------
    # Trend counts
    # ----------------------------------------

    window_1_count = (
        (
            (group["date"] >= start_date)
            &
            (group["date"] <= window_1_end)
        )
        .sum()
    )


    window_2_count = (
        (
            (group["date"] >= window_2_start)
            &
            (group["date"] <= window_2_end)
        )
        .sum()
    )


    window_3_count = (
        (
            (group["date"] >= window_3_start)
            &
            (group["date"] <= window_3_end)
        )
        .sum()
    )


    trend = calculate_trend(
        window_2_count,
        window_3_count
    )


    trend_delta = (
        window_3_count
        - window_2_count
    )


    evidence = (
        select_evidence(
            group,
            n=3
        )
    )


    results.append(
        {
            "cluster_id":
                cluster_id,

            "product_issue":
                issue_name,

            "issue_category":
                category,

            "count":
                total_count,

            "high_count":
                high_count,

            "medium_count":
                medium_count,

            "low_count":
                low_count,

            "priority_score":
                priority_score,

            "window_1_count":
                window_1_count,

            "previous_3d_count":
                window_2_count,

            "recent_3d_count":
                window_3_count,

            "trend_delta":
                trend_delta,

            "trend":
                trend,

            "evidence":
                evidence
        }
    )


# ============================================================
# 9. Rank issues
# ============================================================

result_df = pd.DataFrame(
    results
)


result_df = (
    result_df
    .sort_values(
        [
            "priority_score",
            "count",
            "high_count"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .reset_index(drop=True)
)


result_df.insert(
    0,
    "rank",
    range(
        1,
        len(result_df) + 1
    )
)


# ============================================================
# 10. Save
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. Preview
# ============================================================

print("\n" + "=" * 100)

print(
    "TOP 20 RANKED PRODUCT ISSUES"
)

print("=" * 100)


preview_columns = [
    "rank",
    "product_issue",
    "issue_category",
    "count",
    "high_count",
    "medium_count",
    "low_count",
    "priority_score",
    "previous_3d_count",
    "recent_3d_count",
    "trend"
]


print(
    result_df[
        preview_columns
    ]
    .head(20)
    .to_string(index=False)
)


print("\nSaved to:")

print(
    OUTPUT_FILE
)