# Evaluation Explainer

## Purpose

The evaluation files in this folder document how the system was tested during development.

A separate frozen evaluation set was used so that model performance could be measured against fixed labels rather than judged only from production outputs.

## Evaluation Set

The formal evaluation set contains 150 reviews.

The labels were manually reviewed and frozen using predefined annotation guidelines.

The structured fields include:

- issue category
- specific issue
- sentiment
- severity

The main quantitative evaluation focused on:

- issue-category classification
- High-severity detection

## Issue Taxonomy

The fixed issue categories are:

1. Account & Login
2. Bugs & Technical Issues
3. Performance
4. Feature Problem
5. Feature Request
6. Privacy / Security
7. Other / General

## Models Compared

### Keyword Baseline

The keyword baseline used simple deterministic keyword rules.

Results:

- Accuracy: 0.5733
- Macro F1: 0.3418

The baseline performed reasonably on a small number of explicit keyword-driven categories, but it could not reliably identify:

- Feature Problem
- Feature Request
- Privacy / Security

This showed the limitation of keyword matching for semantic intent.

### Qwen V1

Qwen V1 used a simple category prompt.

Results:

- Accuracy: 0.7067
- Macro F1: 0.5900

V1 improved substantially over the keyword baseline, but error analysis showed strong confusion between Feature Request and Feature Problem.

### Qwen V2

Qwen V2 added clearer category definitions and explicit decision rules.

Results:

- Accuracy: 0.8933
- Macro F1: 0.8398

A major improvement was Feature Request recognition.

Feature Request recall improved from:

`0.1818 → 1.0000`

This improvement came from explicitly distinguishing:

- requests to add, remove, restore, hide, or change a feature;
- existing features that are present but not working.

## High-Severity Evaluation

High-severity detection was evaluated separately because identifying urgent customer problems was an important project outcome.

Final results:

- Precision: 0.7826
- Recall: 0.9000
- F1: 0.8372
- False alarms: 10
- Missed High-severity cases: 4
- Abstentions: 0

The target was at least 85% recall for High-severity feedback.

The final system reached 90% recall.

## Error Analysis

Most High-severity false alarms came from Medium-severity cases rather than Low-severity cases.

This suggests that the model tended to be conservative around the Medium/High boundary.

Medium severity remained the weakest severity class:

- many Medium cases were downgraded to Low;
- some Medium cases were upgraded to High.

This is an important limitation of the current system.

## Evaluation Files

Important files include:

- `formal_baseline_metrics.csv`
- `qwen_v1_formal_metrics.csv`
- `qwen_v2_formal_metrics_normalized.csv`
- `qwen_severity_metrics.csv`
- `qwen_v1_confusion_matrix.csv`
- `qwen_v2_confusion_matrix.csv`
- `qwen_v1_errors.csv`
- `qwen_v2_errors.csv`

These files provide the quantitative results and error analysis used to compare the different versions of the system.