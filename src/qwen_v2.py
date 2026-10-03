import os
import csv
from dotenv import load_dotenv
from openai import OpenAI


# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


# Connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

correct = 0
total = 0
results = []

total_input_tokens = 0
total_output_tokens = 0


SYSTEM_PROMPT = """
You are a customer feedback classifier.

Classify each customer feedback into exactly ONE of these categories:

Payment:
Problems related to checkout, completing an order, payment failure,
refunds, billing, or being unable to finish a purchase.
If the customer is trying to complete an order but is blocked,
redirected back to the cart, or cannot finish checkout, classify it as Payment.

Account:
Problems related to login, password, credentials, account access,
account lock, or authentication.

Performance:
Problems related to slowness, loading time, lag, delay,
or the app taking too long to respond.

Technical:
Problems related to crashes, bugs, errors, unexpected app closure,
or broken technical functionality.

Other:
Feedback that does not clearly belong to the categories above,
including positive feedback or general comments.

Examples:

Feedback:
"I cannot finish my order because the app sends me back to the cart."
Category:
Payment

Feedback:
"The app takes forever to load."
Category:
Performance

Feedback:
"My password is correct but I still cannot get into my account."
Category:
Account

Feedback:
"The app suddenly closes when I open settings."
Category:
Technical

Return ONLY the category name.
"""


with open("data/evaluation_feedback.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        feedback = row["feedback"]
        true_category = row["true_category"]

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": feedback
                }
            ],
            temperature=0
        )

        prediction = response.choices[0].message.content.strip()

        total += 1
        is_correct = prediction == true_category

        if is_correct:
            correct += 1

        if response.usage:
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

        results.append({
            "id": row["id"],
            "feedback": feedback,
            "true_category": true_category,
            "predicted_category": prediction,
            "correct": is_correct
        })

        print("ID:", row["id"])
        print("True:", true_category)
        print("Predicted:", prediction)
        print("-" * 40)


accuracy = correct / total


with open(
    "evaluation/qwen_v2_evaluation_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "id",
        "feedback",
        "true_category",
        "predicted_category",
        "correct"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)


print()
print("Qwen V2 Evaluation Completed")
print("Correct:", correct)
print("Total:", total)
print("Accuracy:", round(accuracy, 2))
print("Input tokens:", total_input_tokens)
print("Output tokens:", total_output_tokens)
print("Total tokens:", total_input_tokens + total_output_tokens)
print("Results saved to evaluation/qwen_v2_evaluation_results.csv")