# Theory3-Modeling-ICU-Outcomes
The table summarizes the variables collected in a study of ICU outcomes. The primary outcome (target) variable is vital status at hospital discharge (STA). Clinicians associated with the study indicate a belief that a key predictor of survival was the patient (AGE).

## Python analysis

`icu_outcomes.py` provides a dependency-free baseline logistic regression using
AGE to predict survival at hospital discharge (STA). The input must be a CSV
with `AGE` and `STA` columns. Missing rows are skipped.

```bash
python icu_outcomes.py path/to/icu_outcomes.csv
```

The default assumes `STA=1` means survival. For text labels, specify the
survival value:

```bash
python icu_outcomes.py path/to/icu_outcomes.csv --positive-label survived
```

The command prints JSON containing the fitted intercept and age coefficient,
accuracy, ROC-AUC, and the test confusion matrix. The default test split is
20% with a reproducible seed of 42.
