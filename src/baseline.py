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


if __name__ == "__main__":

    results = []

    with open("data/sample_feedback.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            feedback = row["feedback"]
            prediction = classify_feedback(feedback)

            results.append({
                "id": row["id"],
                "feedback": feedback,
                "predicted_category": prediction
            })

    with open(
        "data/baseline_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "id",
            "feedback",
            "predicted_category"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("Baseline classification completed.")
    print("Results saved to data/baseline_results.csv")