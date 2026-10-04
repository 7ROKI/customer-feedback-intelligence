"""
Extract structured product intelligence from cleaned customer reviews.

Input:
    data/cleaned_reviews.csv

Output:
    data/all_reviews_structured.csv

For each review, Qwen extracts:
- issue_category
- specific_issue
- sentiment
- severity
- has_specific_issue

The script supports checkpoint/resume behaviour and records token usage.
This module makes paid OpenRouter API calls.
"""

import os
import json
import time
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

INPUT_FILE = "data/cleaned_reviews.csv"

OUTPUT_FILE = "data/all_reviews_structured.csv"


VALID_CATEGORIES = [
    "Account & Login",
    "Bugs & Technical Issues",
    "Performance",
    "Feature Problem",
    "Feature Request",
    "Privacy / Security",
    "Other / General"
]

VALID_SENTIMENTS = [
    "Positive",
    "Neutral",
    "Negative"
]

VALID_SEVERITIES = [
    "High",
    "Medium",
    "Low"
]


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


# ============================================================
# 2. Parse JSON safely
# ============================================================

def parse_json_response(text):

    text = text.strip()

    # Remove markdown fences if the model adds them
    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    return json.loads(text)


# ============================================================
# 3. Structured extraction
# ============================================================

def extract_feedback(feedback):

    prompt = f"""
You are analysing customer feedback for a product management team.

Extract structured information from the customer feedback below.

Use exactly these issue categories:

1. Account & Login
Problems with signup, login, verification, account access,
account bans, blocked phone numbers, passwords, verification codes,
or account recovery.

2. Bugs & Technical Issues
Technical failures such as crashes, freezing, installation failures,
unexpected errors, broken app behaviour, or app-wide malfunction.

3. Performance
Speed, lag, loading, storage usage, responsiveness,
resource consumption, or efficiency problems.

4. Feature Problem
An existing feature is present but does not work correctly.

5. Feature Request
The user wants a feature to be added, removed, restored,
hidden, changed, or redesigned.

6. Privacy / Security
Privacy, fraud, scams, security, unauthorised access,
unsafe behaviour, or exposure of private information.

7. Other / General
General praise, general dissatisfaction, vague comments,
usage statements, unrelated comments, or feedback without
a specific product issue.


SENTIMENT

Choose exactly one:
Positive
Neutral
Negative


SEVERITY

High:
- Cannot complete a core task
- Serious signup/login/account-access failure
- Serious security/privacy issue
- Severe functional or data-loss risk

Medium:
- Important feature is broken or degraded
- Significant experience problem
- App remains generally usable

Low:
- Minor inconvenience
- Feature request or preference
- General praise or dissatisfaction
- No concrete serious product issue


SPECIFIC ISSUE

Write one short, clear phrase describing the PRIMARY product issue.

Examples:
"SMS verification code not received"
"Filters not working"
"Too many advertisements"
"Request to remove Channels"
"App crashes during calls"

If there is no concrete product issue, use a short general description such as:
"General positive feedback"
"General dissatisfaction"
"General usage statement"


HAS SPECIFIC ISSUE

Set has_specific_issue to true only when the feedback describes
a concrete product problem, failure, risk, or actionable request.

Set it to false for:
- general praise
- vague dissatisfaction
- unrelated statements
- general usage statements
- feedback with no actionable product issue


IMPORTANT RULES

- Do not use star rating to infer sentiment or severity.
- Focus on the PRIMARY issue when multiple issues appear.
- If login/signup/account access is blocked, prefer Account & Login.
- If the user asks to add/remove/restore/hide/change something,
  choose Feature Request.
- If an existing feature should work but does not, choose Feature Problem.
- Use Performance only for speed, lag, loading, storage,
  responsiveness, or resource-efficiency issues.
- Do not invent an issue that is not supported by the feedback.


Customer feedback:
{feedback}


Return ONLY valid JSON in exactly this structure:

{{
  "issue_category": "...",
  "specific_issue": "...",
  "sentiment": "...",
  "severity": "...",
  "has_specific_issue": true
}}
""".strip()


    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )


    raw_output = response.choices[0].message.content.strip()

    data = parse_json_response(raw_output)


    # ========================================================
    # Validate output
    # ========================================================

    if data.get("issue_category") not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category: {data.get('issue_category')}"
        )

    if data.get("sentiment") not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment: {data.get('sentiment')}"
        )

    if data.get("severity") not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity: {data.get('severity')}"
        )

    if not isinstance(
        data.get("has_specific_issue"),
        bool
    ):
        raise ValueError(
            "has_specific_issue is not a boolean"
        )

    if not data.get("specific_issue"):
        raise ValueError(
            "specific_issue is empty"
        )


    usage = response.usage

    input_tokens = (
        usage.prompt_tokens
        if usage
        else 0
    )

    output_tokens = (
        usage.completion_tokens
        if usage
        else 0
    )

    total_tokens = (
        usage.total_tokens
        if usage
        else 0
    )


    return (
        data,
        input_tokens,
        output_tokens,
        total_tokens
    )


# ============================================================
# 4. Load cleaned reviews
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print("====================================")
print("FULL STRUCTURED EXTRACTION")
print("====================================")

print("Total cleaned reviews:", len(df))
print("Model:", MODEL)


# ============================================================
# 5. Resume support
# ============================================================

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_csv(
        OUTPUT_FILE,
        dtype=str
    )

    completed_ids = set(
        existing["review_id"]
    )

    print(
        "Existing completed reviews:",
        len(existing)
    )

else:

    existing = pd.DataFrame()

    completed_ids = set()

    print(
        "No existing output found. Starting fresh."
    )


# ============================================================
# 6. Run extraction
# ============================================================

new_results = []


for index, row in df.iterrows():

    review_id = row["review_id"]

    if review_id in completed_ids:
        continue


    feedback = row["feedback"]

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"Review ID: {review_id}"
    )


    try:

        (
            structured,
            input_tokens,
            output_tokens,
            total_tokens
        ) = extract_feedback(feedback)


        print(
            "Category:",
            structured["issue_category"]
        )

        print(
            "Specific issue:",
            structured["specific_issue"]
        )

        print(
            "Severity:",
            structured["severity"]
        )


        result = {
            "review_id": review_id,
            "feedback": feedback,
            "rating": row.get("rating", ""),
            "date": row.get("date", ""),
            "app_id": row.get("app_id", ""),

            "issue_category":
                structured["issue_category"],

            "specific_issue":
                structured["specific_issue"],

            "sentiment":
                structured["sentiment"],

            "severity":
                structured["severity"],

            "has_specific_issue":
                structured["has_specific_issue"],

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                total_tokens
        }


        new_results.append(result)


        # ====================================================
        # Save checkpoint after every successful review
        # ====================================================

        new_df = pd.DataFrame(
            new_results
        )


        if len(existing) > 0:

            combined = pd.concat(
                [
                    existing,
                    new_df
                ],
                ignore_index=True
            )

        else:

            combined = new_df


        combined.to_csv(
            OUTPUT_FILE,
            index=False
        )


        # Small delay for stability
        time.sleep(0.2)


    except Exception as e:

        print("\n====================================")
        print("STOPPED")
        print("====================================")

        print(
            "Review ID:",
            review_id
        )

        print(
            "Feedback:",
            feedback
        )

        print(
            "Error:",
            e
        )

        print(
            "\nCompleted results were saved."
        )

        print(
            "Do NOT immediately rerun if this was "
            "a model-output problem."
        )

        break


# ============================================================
# 7. Final summary
# ============================================================

if os.path.exists(OUTPUT_FILE):

    results = pd.read_csv(
        OUTPUT_FILE
    )

    print("\n====================================")
    print("EXTRACTION SUMMARY")
    print("====================================")

    print(
        "Completed reviews:",
        len(results),
        "/",
        len(df)
    )


    if "input_tokens" in results.columns:

        total_input_tokens = (
            pd.to_numeric(
                results["input_tokens"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        total_output_tokens = (
            pd.to_numeric(
                results["output_tokens"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        total_tokens = (
            pd.to_numeric(
                results["total_tokens"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )


        print("\n=== Token Usage ===")

        print(
            "Input tokens:",
            int(total_input_tokens)
        )

        print(
            "Output tokens:",
            int(total_output_tokens)
        )

        print(
            "Total tokens:",
            int(total_tokens)
        )


    if len(results) == len(df):

        print(
            "\nAll 758 reviews completed successfully!"
        )

        print(
            "Output saved to:"
        )

        print(
            OUTPUT_FILE
        )

    else:

        print(
            "\nExtraction is incomplete."
        )

        print(
            "Check the error above before continuing."
        )