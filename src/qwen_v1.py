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
                    "content": """
You are a customer feedback classifier.

Classify the customer feedback into exactly one of these categories:

Payment
Account
Performance
Technical
Other

Return only the category name.
"""
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
    "evaluation/qwen_v1_evaluation_results.csv",
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
print("Qwen V1 Evaluation Completed")
print("Correct:", correct)
print("Total:", total)
print("Accuracy:", round(accuracy, 2))
print("Input tokens:", total_input_tokens)
print("Output tokens:", total_output_tokens)
print("Total tokens:", total_input_tokens + total_output_tokens)
print("Results saved to evaluation/qwen_v1_evaluation_results.csv")