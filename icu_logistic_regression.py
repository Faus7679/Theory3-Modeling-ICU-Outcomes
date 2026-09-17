"""Demonstrate the STA ~ AGE logistic regression assignment.

Run without arguments to use the small illustrative data set below, or pass a
CSV file containing numeric AGE and binary STA columns:

    python icu_logistic_regression.py patient_data.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


# The repository does not include patient-level data. These values are only an
# illustrative data set so that every assignment result can be reproduced.
EXAMPLE_DATA = [
    (25, 1), (30, 1), (35, 1), (40, 1), (45, 1),
    (50, 1), (55, 1), (60, 0), (65, 0), (70, 0),
    (75, 0), (80, 0),
]


def read_data(path: str | None) -> list[tuple[float, int]]:
    if path is None:
        return EXAMPLE_DATA.copy()

    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        if not rows.fieldnames or not {"AGE", "STA"} <= set(rows.fieldnames):
            raise ValueError("CSV must contain AGE and STA columns.")
        data = [(float(row["AGE"]), int(row["STA"])) for row in rows]

    if not data:
        raise ValueError("CSV does not contain any observations.")
    return data


def logistic(linear_predictor: float) -> float:
    """Return 1 / (1 + exp(-linear_predictor)) without overflow."""
    if linear_predictor >= 0:
        return 1 / (1 + math.exp(-linear_predictor))
    exponent = math.exp(linear_predictor)
    return exponent / (1 + exponent)


def fit_logistic_regression(data: Iterable[tuple[float, int]]) -> tuple[float, float]:
    """Fit STA ~ AGE with Newton-Raphson updates."""
    observations = list(data)
    intercept, slope = 0.0, 0.0

    for _ in range(100):
        gradient_0 = gradient_1 = hessian_00 = hessian_01 = hessian_11 = 0.0
        for age, status in observations:
            probability = logistic(intercept + slope * age)
            weight = probability * (1 - probability)
            residual = status - probability
            gradient_0 += residual
            gradient_1 += residual * age
            hessian_00 -= weight
            hessian_01 -= weight * age
            hessian_11 -= weight * age * age

        determinant = hessian_00 * hessian_11 - hessian_01**2
        if abs(determinant) < 1e-12:
            raise ValueError("The model could not be fitted; check the input data.")

        # Solve H * delta = gradient, then update beta <- beta - H^-1 gradient.
        delta_0 = (hessian_11 * gradient_0 - hessian_01 * gradient_1) / determinant
        delta_1 = (-hessian_01 * gradient_0 + hessian_00 * gradient_1) / determinant
        intercept -= delta_0
        slope -= delta_1
        if max(abs(delta_0), abs(delta_1)) < 1e-10:
            return intercept, slope

    raise ValueError("The model did not converge after 100 iterations.")


def logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and explain STA ~ AGE.")
    parser.add_argument("csv_file", nargs="?", help="CSV file with AGE and STA columns")
    args = parser.parse_args()
    data = read_data(args.csv_file)
    intercept, slope = fit_logistic_regression(data)
    odds_ratio = math.exp(slope)
    ages = [age for age, _ in data]

    print(f"Observations: {len(data)}")
    print("\n1. Fitted logistic regression model")
    print(f"   beta_0 (intercept) = {intercept:.6f}")
    print(f"   beta_1 (AGE slope) = {slope:.6f}")
    print(
        "   P(STA=1 | AGE) = "
        f"1 / (1 + exp(-({intercept:.6f} + {slope:.6f} * AGE)))"
    )

    print("\n2. Logit transformation")
    print(f"   logit(P(STA=1 | AGE)) = {intercept:.6f} + {slope:.6f} * AGE")

    print("\n3. Age coefficient interpretation")
    direction = "decreases" if odds_ratio < 1 else "increases"
    percent = abs(odds_ratio - 1) * 100
    print(f"   Odds ratio exp(beta_1) = {odds_ratio:.6f}")
    print(f"   Each additional year of age {direction} the event odds by {percent:.2f}%.")

    print("\n4. Predicted results")
    print("   AGE       probability STA=1       logit")
    for age in sorted(set(ages)):
        probability = logistic(intercept + slope * age)
        print(f"   {age:>3.0f}       {probability:>9.4f}             {logit(probability):>8.4f}")


if __name__ == "__main__":
    main()
