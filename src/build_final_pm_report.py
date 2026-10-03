import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "data/ranked_product_issues.csv"

OUTPUT_FILE = "data/final_pm_issue_report.csv"

TOP_N = 15


# ============================================================
# 2. Load ranked issues
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Total ranked clusters:", len(df))


# ============================================================
# 3. Keep top issues for PM-facing report
# ============================================================

report = df.head(TOP_N).copy()


# ============================================================
# 4. Build readable severity mix
# ============================================================

def severity_mix(row):

    parts = []

    if row["high_count"] > 0:
        parts.append(
            f'{int(row["high_count"])} High'
        )

    if row["medium_count"] > 0:
        parts.append(
            f'{int(row["medium_count"])} Medium'
        )

    if row["low_count"] > 0:
        parts.append(
            f'{int(row["low_count"])} Low'
        )

    return ", ".join(parts)


report["severity_mix"] = report.apply(
    severity_mix,
    axis=1
)


# ============================================================
# 5. Human-readable trend
# ============================================================

def trend_description(row):

    previous = int(
        row["previous_3d_count"]
    )

    recent = int(
        row["recent_3d_count"]
    )

    trend = row["trend"]

    if trend == "New":
        return (
            f"New ({recent} recent mentions)"
        )

    if trend == "Rising":
        return (
            f"Rising ({previous} → {recent})"
        )

    if trend == "Falling":
        return (
            f"Falling ({previous} → {recent})"
        )

    return (
        f"Stable ({previous} → {recent})"
    )


report["trend_summary"] = report.apply(
    trend_description,
    axis=1
)


# ============================================================
# 6. PM recommendation engine
# ============================================================

def pm_recommendation(row):

    category = row["issue_category"]

    high = int(row["high_count"])
    medium = int(row["medium_count"])
    low = int(row["low_count"])

    trend = row["trend"]
    count = int(row["count"])

    high_ratio = (
        high / count
        if count > 0
        else 0
    )


    # --------------------------------------------------------
    # Privacy / Security receives escalation attention
    # --------------------------------------------------------

    if category == "Privacy / Security":

        return (
            "Escalate for privacy/security review"
        )


    # --------------------------------------------------------
    # Feature requests should be evaluated for roadmap value
    # rather than treated like operational incidents
    # --------------------------------------------------------

    if category == "Feature Request":

        if count >= 5:

            return (
                "Validate demand for roadmap consideration"
            )

        return (
            "Monitor request frequency"
        )


    # --------------------------------------------------------
    # High-severity issue affecting a substantial share
    # --------------------------------------------------------

    if (
        high_ratio >= 0.5
        and trend in ["Rising", "New"]
    ):

        return (
            "Immediate investigation"
        )


    # --------------------------------------------------------
    # Significant high-severity presence
    # --------------------------------------------------------

    if high >= 2:

        return (
            "High-priority root-cause analysis"
        )


    # --------------------------------------------------------
    # Mostly medium-severity issue that is emerging/rising
    # --------------------------------------------------------

    if (
        medium > 0
        and trend in ["Rising", "New"]
    ):

        return (
            "Investigate possible product regression"
        )


    # --------------------------------------------------------
    # Low-impact declining issue
    # --------------------------------------------------------

    if (
        low > 0
        and trend == "Falling"
    ):

        return (
            "Monitor; no immediate action"
        )


    return (
        "Monitor and review supporting evidence"
    )
report["pm_recommendation"] = (
    report.apply(
        pm_recommendation,
        axis=1
    )
)

# ============================================================
# 7. Recommended next step
# ============================================================

def recommended_next_step(row):

    issue = str(
        row["product_issue"]
    ).lower()

    category = row[
        "issue_category"
    ]


    # --------------------------------------------------------
    # Account & Login
    # --------------------------------------------------------

    if category == "Account & Login":

        if (
            "verification" in issue
            or "code" in issue
        ):

            return (
                "Check verification delivery failures, "
                "retry logic, and affected carriers or regions."
            )

        if (
            "ban" in issue
            or "blocked" in issue
        ):

            return (
                "Audit ban reasons, recovery options, "
                "appeal flow, and support response time."
            )

        if (
            "register" in issue
            or "create" in issue
            or "sign up" in issue
        ):

            return (
                "Review signup funnel drop-off points, "
                "validation errors, and onboarding requirements."
            )

        return (
            "Inspect login failure logs, affected versions, "
            "devices, and account-access funnel steps."
        )


    # --------------------------------------------------------
    # Bugs & Technical Issues
    # --------------------------------------------------------

    if category == "Bugs & Technical Issues":

        if (
            "install" in issue
            or "update" in issue
        ):

            return (
                "Check installation/update failures by "
                "app version, OS version, and device type."
            )

        if (
            "call" in issue
        ):

            return (
                "Review call failure logs and identify "
                "device, network, or version-specific patterns."
            )

        return (
            "Review crash/error logs and reproduce the "
            "failure on affected versions and devices."
        )


    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    if category == "Performance":

        if (
            "battery" in issue
            or "storage" in issue
        ):

            return (
                "Profile resource usage and compare battery, "
                "storage, and background activity by version."
            )

        return (
            "Compare startup, loading, and responsiveness "
            "metrics across recent app versions."
        )


    # --------------------------------------------------------
    # Feature Problem
    # --------------------------------------------------------

    if category == "Feature Problem":

        return (
            "Reproduce the affected feature workflow and "
            "compare failures across versions and devices."
        )


    # --------------------------------------------------------
    # Feature Request
    # --------------------------------------------------------

    if category == "Feature Request":

        return (
            "Review request frequency, affected user segments, "
            "and current feature engagement before roadmap action."
        )


    # --------------------------------------------------------
    # Privacy / Security
    # --------------------------------------------------------

    if category == "Privacy / Security":

        return (
            "Escalate to the relevant privacy/security owner "
            "and review evidence for immediate risk assessment."
        )


    return (
        "Review the supporting customer evidence and "
        "monitor whether mentions increase."
    )


report["recommended_next_step"] = (
    report.apply(
        recommended_next_step,
        axis=1
    )
)


# ============================================================
# 8. Why this matters
# ============================================================

def why_it_matters(row):

    category = row[
        "issue_category"
    ]

    high = int(
        row["high_count"]
    )

    count = int(
        row["count"]
    )

    trend = row[
        "trend"
    ]


    if high == count and count > 0:

        severity_text = (
            "All observed cases are high severity"
        )

    elif high > 0:

        severity_text = (
            f"{high} of {count} cases are high severity"
        )

    else:

        severity_text = (
            "No high-severity cases in this cluster"
        )


    if trend == "Rising":

        trend_text = (
            "mentions are increasing"
        )

    elif trend == "New":

        trend_text = (
            "the issue appeared in the latest period"
        )

    elif trend == "Falling":

        trend_text = (
            "mentions are decreasing"
        )

    else:

        trend_text = (
            "mentions are stable"
        )


    return (
        f"{severity_text}; {trend_text}. "
        f"Category: {category}."
    )


report["why_it_matters"] = (
    report.apply(
        why_it_matters,
        axis=1
    )
)


# ============================================================
# 9. Select PM-facing columns
# ============================================================

final_report = report[
    [
        "rank",
        "product_issue",
        "issue_category",
        "count",
        "priority_score",
        "severity_mix",
        "trend_summary",
        "why_it_matters",
        "pm_recommendation",
        "recommended_next_step",
        "evidence",
        "cluster_id"
    ]
].copy()


final_report.columns = [
    "Rank",
    "Product Issue",
    "Category",
    "Mentions",
    "Priority Score",
    "Severity Mix",
    "Trend",
    "Why It Matters",
    "PM Recommendation",
    "Recommended Next Step",
    "Customer Evidence",
    "Cluster ID"
]


# ============================================================
# 10. Save
# ============================================================

final_report.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. Preview
# ============================================================

print("\n" + "=" * 120)

print(
    "FINAL PM DECISION-SUPPORT REPORT"
)

print("=" * 120)


preview_columns = [
    "Rank",
    "Product Issue",
    "Mentions",
    "Priority Score",
    "Trend",
    "PM Recommendation"
]


print(
    final_report[
        preview_columns
    ]
    .to_string(
        index=False
    )
)


print("\nSaved to:")

print(
    OUTPUT_FILE
)