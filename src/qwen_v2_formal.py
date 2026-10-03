import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


def classify_with_qwen_v2(feedback):

    prompt = f"""
You are classifying customer feedback for a product team.

Choose exactly ONE category from the seven categories below.

CATEGORY DEFINITIONS

1. Account & Login
Problems that prevent or interfere with signing up, logging in,
account verification, account access, bans, blocked phone numbers,
passwords, verification codes, or account recovery.

2. Bugs & Technical Issues
Technical failures such as crashes, freezing, installation failures,
unexpected errors, broken app behavior, or the app failing to open.

3. Performance
Speed or resource problems such as lag, slowness, loading delays,
high storage usage, excessive resource consumption, or poor efficiency.

4. Feature Problem
An EXISTING feature is present but does not work correctly.
Examples:
- filters not working
- notifications not appearing
- messages not syncing
- camera roll permission not recognized

5. Feature Request
The user wants a feature to be ADDED, REMOVED, RESTORED, HIDDEN,
CHANGED, or redesigned.
Examples:
- please remove Channels
- bring back the old dark theme
- add a mute stories option
- restore the Lite version

6. Privacy / Security
Privacy, security, scams, fraud, unauthorized access, unsafe behavior,
data exposure, or privacy expectations being violated.

7. Other / General
General praise, general dissatisfaction, vague comments,
usage statements, unrelated comments, or feedback that does not
describe a specific product issue.

DECISION RULES

- If the user cannot sign up, log in, verify, recover, or access
  their account, choose Account & Login even if an error or bug
  causes the problem.

- If the user asks to add, remove, restore, hide, or change a feature,
  choose Feature Request.

- If an existing feature should work but currently does not work,
  choose Feature Problem.

- Use Bugs & Technical Issues for general technical failures such
  as crashes, freezing, installation errors, or app-wide malfunction.

- Use Performance only for speed, lag, loading, storage,
  responsiveness, or resource-efficiency problems.

- Use Privacy / Security when the main concern is fraud, scams,
  security, privacy, unsafe access, or exposure of private data.

- Do not infer a specific issue from vague praise or dissatisfaction.
  Use Other / General instead.

- Focus on the PRIMARY user problem when multiple issues are mentioned.

Customer feedback:
{feedback}

Return only the exact category name.
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

    prediction = response.choices[0].message.content.strip()

    usage = response.usage

    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    return (
        prediction,
        input_tokens,
        output_tokens,
        total_tokens
    )