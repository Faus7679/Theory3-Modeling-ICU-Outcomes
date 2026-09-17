"""Train and evaluate an age-only logistic model for ICU discharge status.

The input CSV must contain:
    AGE: patient age in years
    STA: vital status at hospital discharge

STA may be numeric (0/1) or textual. For textual values, ``--positive-label``
defines the value representing survival.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Dataset:
    ages: list[float]
    outcomes: list[int]


@dataclass
class LogisticModel:
    intercept: float
    coefficient: float

    def predict_probability(self, age: float) -> float:
        score = self.intercept + self.coefficient * age
        score = max(-700.0, min(700.0, score))
        return 1.0 / (1.0 + math.exp(-score))


def _parse_outcome(value: str, positive_label: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "alive", "survived", positive_label.lower()}:
        return 1
    if normalized in {"0", "false", "no", "dead", "died"}:
        return 0
    raise ValueError(
        f"Unsupported STA value {value!r}; use 0/1 or pass "
        f"--positive-label with the survival value."
    )


def load_dataset(path: Path, positive_label: str) -> Dataset:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "AGE" not in reader.fieldnames or "STA" not in reader.fieldnames:
            raise ValueError("Input CSV must contain AGE and STA columns.")

        ages: list[float] = []
        outcomes: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            age_value = (row.get("AGE") or "").strip()
            status_value = (row.get("STA") or "").strip()
            if not age_value or not status_value:
                continue
            try:
                age = float(age_value)
                if not 0 <= age <= 120:
                    raise ValueError
                outcome = _parse_outcome(status_value, positive_label)
            except ValueError as error:
                raise ValueError(f"Invalid AGE/STA value on CSV line {line_number}.") from error
            ages.append(age)
            outcomes.append(outcome)

    if len(ages) < 4:
        raise ValueError("At least four complete observations are required.")
    if len(set(outcomes)) < 2:
        raise ValueError("STA must contain both outcome classes.")
    return Dataset(ages, outcomes)


def split_dataset(data: Dataset, test_size: float, seed: int) -> tuple[Dataset, Dataset]:
    indices = list(range(len(data.ages)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, min(len(indices) - 1, round(len(indices) * test_size)))
    test_indices = set(indices[:test_count])
    train = [index for index in indices if index not in test_indices]
    test = [index for index in indices if index in test_indices]
    return (
        Dataset([data.ages[i] for i in train], [data.outcomes[i] for i in train]),
        Dataset([data.ages[i] for i in test], [data.outcomes[i] for i in test]),
    )


def fit_logistic_model(data: Dataset, epochs: int = 5000, learning_rate: float = 0.01) -> LogisticModel:
    mean_age = sum(data.ages) / len(data.ages)
    scale = math.sqrt(sum((age - mean_age) ** 2 for age in data.ages) / len(data.ages)) or 1.0
    intercept = 0.0
    coefficient = 0.0

    for _ in range(epochs):
        gradient_intercept = 0.0
        gradient_coefficient = 0.0
        for age, outcome in zip(data.ages, data.outcomes):
            standardized_age = (age - mean_age) / scale
            probability = 1.0 / (1.0 + math.exp(-(intercept + coefficient * standardized_age)))
            error = probability - outcome
            gradient_intercept += error
            gradient_coefficient += error * standardized_age
        divisor = len(data.ages)
        intercept -= learning_rate * gradient_intercept / divisor
        coefficient -= learning_rate * gradient_coefficient / divisor

    # Store parameters in the original AGE units for interpretable predictions.
    return LogisticModel(
        intercept - coefficient * mean_age / scale,
        coefficient / scale,
    )


def _accuracy(actual: Iterable[int], predicted: Iterable[int]) -> float:
    actual_values = list(actual)
    predicted_values = list(predicted)
    return sum(a == p for a, p in zip(actual_values, predicted_values)) / len(actual_values)


def _roc_auc(actual: list[int], probabilities: list[float]) -> float | None:
    positives = [score for label, score in zip(actual, probabilities) if label == 1]
    negatives = [score for label, score in zip(actual, probabilities) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(positive > negative for positive in positives for negative in negatives)
    ties = sum(positive == negative for positive in positives for negative in negatives)
    return (wins + ties / 2) / (len(positives) * len(negatives))


def evaluate(model: LogisticModel, data: Dataset, threshold: float) -> dict[str, float | int | None]:
    probabilities = [model.predict_probability(age) for age in data.ages]
    predicted = [int(probability >= threshold) for probability in probabilities]
    return {
        "observations": len(data.ages),
        "accuracy": _accuracy(data.outcomes, predicted),
        "roc_auc": _roc_auc(data.outcomes, probabilities),
        "true_negatives": sum(a == 0 and p == 0 for a, p in zip(data.outcomes, predicted)),
        "false_positives": sum(a == 0 and p == 1 for a, p in zip(data.outcomes, predicted)),
        "false_negatives": sum(a == 1 and p == 0 for a, p in zip(data.outcomes, predicted)),
        "true_positives": sum(a == 1 and p == 1 for a, p in zip(data.outcomes, predicted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV file containing AGE and STA columns")
    parser.add_argument("--positive-label", default="1", help="STA value representing survival")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.threshold < 1:
        parser.error("--test-size and --threshold must be between 0 and 1.")

    dataset = load_dataset(args.csv_path, args.positive_label)
    train, test = split_dataset(dataset, args.test_size, args.seed)
    model = fit_logistic_model(train)
    result = {
        "model": {"intercept": model.intercept, "age_coefficient": model.coefficient},
        "train": evaluate(model, train, args.threshold),
        "test": evaluate(model, test, args.threshold),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
