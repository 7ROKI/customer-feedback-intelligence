"""
Calculate recorded API cost and production cost scalability.

The script uses measured token usage from formal evaluation
and production runs to calculate:

- cost per run
- total recorded API cost
- average production cost per review
- approximate review capacity under a fixed $10 budget

No API call is made in this module.
"""
# ============================================================
# Cost Analysis
# Qwen3 30B A3B Instruct 2507
# ============================================================

INPUT_PRICE_PER_MILLION = 0.04815
OUTPUT_PRICE_PER_MILLION = 0.1931


runs = {
    "Qwen V1 formal evaluation": {
        "reviews": 150,
        "input_tokens": 12290,
        "output_tokens": 527
    },

    "Qwen V2 formal evaluation": {
        "reviews": 150,
        "input_tokens": 78590,
        "output_tokens": 503
    },

    "Severity formal evaluation": {
        "reviews": 150,
        "input_tokens": 37790,
        "output_tokens": 228
    },

    "Final structured extraction": {
        "reviews": 758,
        "input_tokens": 480695,
        "output_tokens": 36020
    }
}


def calculate_cost(
    input_tokens,
    output_tokens
):

    input_cost = (
        input_tokens
        / 1_000_000
        * INPUT_PRICE_PER_MILLION
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * OUTPUT_PRICE_PER_MILLION
    )

    return (
        input_cost,
        output_cost,
        input_cost + output_cost
    )


print("=" * 80)
print("API COST ANALYSIS")
print("=" * 80)


total_input = 0
total_output = 0
total_cost = 0


for name, run in runs.items():

    input_cost, output_cost, cost = (
        calculate_cost(
            run["input_tokens"],
            run["output_tokens"]
        )
    )

    total_input += run["input_tokens"]
    total_output += run["output_tokens"]
    total_cost += cost

    print(f"\n{name}")

    print(
        "Reviews:",
        run["reviews"]
    )

    print(
        "Input tokens:",
        run["input_tokens"]
    )

    print(
        "Output tokens:",
        run["output_tokens"]
    )

    print(
        f"Estimated cost: ${cost:.6f}"
    )


print("\n" + "=" * 80)
print("TOTAL RECORDED FORMAL RUNS")
print("=" * 80)

print(
    "Input tokens:",
    total_input
)

print(
    "Output tokens:",
    total_output
)

print(
    f"Approximate total API cost: "
    f"${total_cost:.6f}"
)


# ============================================================
# Production cost per review
# Use the final 758-review extraction because this represents
# the actual production pipeline.
# ============================================================

production = runs[
    "Final structured extraction"
]

avg_input = (
    production["input_tokens"]
    / production["reviews"]
)

avg_output = (
    production["output_tokens"]
    / production["reviews"]
)


_, _, cost_per_review = (
    calculate_cost(
        avg_input,
        avg_output
    )
)


budget = 10

reviews_for_10_dollars = (
    budget
    / cost_per_review
)


print("\n" + "=" * 80)
print("PRODUCTION COST")
print("=" * 80)

print(
    f"Average input tokens/review: "
    f"{avg_input:.2f}"
)

print(
    f"Average output tokens/review: "
    f"{avg_output:.2f}"
)

print(
    f"Estimated API cost/review: "
    f"${cost_per_review:.8f}"
)

print(
    f"Estimated reviews for $10: "
    f"{reviews_for_10_dollars:,.0f}"
)


print("\nNote:")
print(
    "BGE-M3 embedding, clustering, ranking, "
    "trend calculation, and PM recommendation "
    "run locally and do not add OpenRouter token cost."
)