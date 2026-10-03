import csv


def classify_feedback(feedback):
    text = feedback.lower()

    if "payment" in text or "checkout" in text or "refund" in text:
        return "Payment"

    elif "login" in text or "log in" in text or "password" in text or "account" in text:
        return "Account"

    elif "slow" in text or "loading" in text or "lag" in text:
        return "Performance"

    elif "crash" in text or "error" in text or "bug" in text:
        return "Technical"

    else:
        return "Other"


correct = 0
total = 0
results = []

with open("data/evaluation_feedback.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        feedback = row["feedback"]
        true_category = row["true_category"]

        predicted_category = classify_feedback(feedback)

        total += 1
        is_correct = predicted_category == true_category

        if is_correct:
            correct += 1

        results.append({
            "id": row["id"],
            "feedback": feedback,
            "true_category": true_category,
            "predicted_category": predicted_category,
            "correct": is_correct
        })


accuracy = correct / total


with open(
    "evaluation/baseline_evaluation_results.csv",
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


print("Correct:", correct)
print("Total:", total)
print("Accuracy:", round(accuracy, 2))
print("Results saved to evaluation/baseline_evaluation_results.csv")