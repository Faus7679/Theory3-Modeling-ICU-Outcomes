# Theory3-Modeling-ICU-Outcomes

The table summarizes variables collected in a study of ICU outcomes. The primary outcome variable is vital status at hospital discharge (`STA`), and a clinically important predictor is patient age (`AGE`).

## Why a logistic regression model is appropriate

A logistic regression model is appropriate because `STA` is a **binary outcome**. Vital status at discharge has only two possible categories, such as survived versus did not survive. Linear regression is not appropriate for this kind of response because it assumes a continuous outcome and can produce predicted values below 0 or above 1, which are not valid probabilities. In contrast, logistic regression models the **probability** of the event and constrains fitted values to the interval from 0 to 1.

Let

\[
\pi(\text{AGE}) = P(\text{STA} = 1 \mid \text{AGE})
\]

denote the probability that a patient is in the event category coded as 1 for `STA` at a given age.

## 1. Logistic regression model of STA on AGE

The logistic regression model relating hospital discharge status (`STA`) to age (`AGE`) is

\[
\pi(\text{AGE}) = \frac{\exp(\beta_0 + \beta_1 \text{AGE})}{1 + \exp(\beta_0 + \beta_1 \text{AGE})}.
\]

In this model:

- \(\beta_0\) is the intercept, representing the baseline log-odds when `AGE = 0`.
- \(\beta_1\) is the slope for age, representing the change in the log-odds of the event for a one-year increase in age.

Equivalently, the model may be written as

\[
P(\text{STA} = 1 \mid \text{AGE}) = \frac{1}{1 + \exp[-(\beta_0 + \beta_1 \text{AGE})]}.
\]

If \(\beta_1 < 0\), increasing age is associated with a lower probability of survival; if \(\beta_1 > 0\), increasing age is associated with a higher probability of survival.

## 2. Logit transformation of the logistic regression model

The logit transformation converts the probability of the event into the log of the odds:

\[
\text{logit}[\pi(\text{AGE})] = \log\left(\frac{\pi(\text{AGE})}{1-\pi(\text{AGE})}\right).
\]

For the model of `STA` on `AGE`, the logit equation is

\[
\log\left(\frac{P(\text{STA}=1 \mid \text{AGE})}{1 - P(\text{STA}=1 \mid \text{AGE})}\right) = \beta_0 + \beta_1 \text{AGE}.
\]

This linear form is important because it shows that logistic regression models a straight-line relationship between `AGE` and the **log-odds** of the discharge outcome, even though the probability itself changes nonlinearly.

## Python demonstration

`icu_logistic_regression.py` fits the `STA ~ AGE` model and prints the fitted
equation, logit equation, odds ratio, coefficient interpretation, and predicted
probabilities. The repository does not contain patient-level data, so running
the script without an argument uses a clearly labeled illustrative data set:

```bash
python icu_logistic_regression.py
```

For actual data, provide a CSV with numeric `AGE` and binary `STA` columns:

```bash
python icu_logistic_regression.py patient_data.csv
```

## Interpretation of the age coefficient

Exponentiating the coefficient gives an odds ratio:

\[
\exp(\beta_1).
\]

This quantity represents the multiplicative change in the odds of the event coded as 1 for each additional year of age. For example, if \(\exp(\beta_1)=0.95\), then each one-year increase in age is associated with a 5% decrease in the odds of the event.

## References

- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley.
- Kleinbaum, D. G., & Klein, M. (2010). *Logistic regression: A self-learning text* (3rd ed.). Springer.
