# Data Explainer

## Source

This project uses the Hugging Face dataset:

`UniqueData/messengers-reviews-google-play`

The original dataset contains 6,000 Google Play reviews across six messaging applications and five languages.

## Data Cleaning

The production dataset was created using the following deterministic cleaning rules:

1. Keep English-language reviews only.
2. Remove reviews with missing feedback text.
3. Remove duplicated review text.
4. Strip leading and trailing whitespace.
5. Remove reviews shorter than 10 characters.
6. Keep only the fields needed for the project:
   - review ID
   - feedback text
   - rating
   - date
   - app ID

After cleaning, 758 reviews remained.

## Main Data Files

### `cleaned_reviews.csv`

Contains the 758 cleaned reviews used as input to the production pipeline.

### `all_reviews_structured.csv`

Contains the structured Qwen output for all 758 reviews.

Each review includes:

- issue category
- specific issue
- sentiment
- severity
- whether the review contains a specific actionable issue

### `specific_issues_for_clustering.csv`

Contains reviews identified as having specific product issues and prepared for semantic clustering.

### `clustered_specific_issues.csv`

Contains the actionable issue records after BGE-M3 semantic embedding and category-aware clustering.

### `ranked_product_issues.csv`

Contains cluster-level product intelligence, including:

- product issue
- issue category
- number of mentions
- severity counts
- priority score
- recent trend
- customer evidence

### `final_pm_issue_report.csv`

Contains the final PM-facing decision-support output.

It includes:

- ranked product issues
- mentions
- severity mix
- trend
- why the issue matters
- PM recommendation
- recommended next step
- customer evidence

### `formal_evaluation_inputs.csv`

Contains the 150 frozen evaluation inputs used for formal evaluation.

### `formal_ground_truth.csv`

Contains the corresponding frozen evaluation labels.

## Important Notes

Star rating was retained as source metadata, but it was not used to determine sentiment or severity.

The production dataset and the formal evaluation files were kept separate so that evaluation could be performed against frozen labels rather than against production outputs.