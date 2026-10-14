# Falcon 9: chronological validation

An independently rerun educational analysis of 90 historical missions.
Train: **2010-06-04 to 2020-01-07 (72 missions)**.
Holdout: **2020-01-19 to 2020-11-05 (18 missions)**,
with **15 successful and 3 unsuccessful labels**.

Selected using only training cross-validation: **Decision Tree**. All candidates' holdout
metrics below are descriptive; do not select a new winner using this test set.

| Model | CV balanced accuracy | Holdout accuracy | Balanced accuracy | Precision | Recall | F1 | TN / FP / FN / TP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Majority baseline | 50.0% | 83.3% | 50.0% | 83.3% | 100.0% | 90.9% | 0 / 3 / 0 / 15 |
| Logistic Regression | 61.1% | 88.9% | 66.7% | 88.2% | 100.0% | 93.8% | 1 / 2 / 0 / 15 |
| SVM | 66.5% | 88.9% | 66.7% | 88.2% | 100.0% | 93.8% | 1 / 2 / 0 / 15 |
| Decision Tree | 70.1% | 88.9% | 66.7% | 88.2% | 100.0% | 93.8% | 1 / 2 / 0 / 15 |
| KNN | 67.8% | 88.9% | 66.7% | 88.2% | 100.0% | 93.8% | 1 / 2 / 0 / 15 |

## Method and limits

- Three expanding training windows; each validation date is later than every training date. Windows contain similar mission counts, not equal calendar durations.
- Imputation, scaling and categorical encoding are fitted inside each training fold. Unseen categories are handled without fitting on test data.
- The saved course CSV already mean-imputes PayloadMass across the full dataset. PayloadMass is excluded here, rather than treating its imputed values as raw measurements.
- Outcome and Class are excluded from inputs. LandingPad, Serial, ReusedCount and Flights are also excluded because their historical availability is not established. FlightNumber is an era proxy; configuration fields remain retrospective course records, not audited pre-launch inputs.
- Class 0 combines unsuccessful landings and missions without a successful recorded recovery. This is not a conditional probability of success given an attempted landing.
- The small, success-heavy holdout can make a majority baseline look strong. Balanced accuracy measures both classes equally; precision and recall use success (Class 1) as positive.
- CV scores are tuning scores, not an unbiased nested-CV estimate. No score is a current-launch prediction or operational validation. Once inspected, this holdout is consumed; future iteration needs a fresh later dataset.
- No accuracy confidence interval is presented: launches change over time and repeated boosters are not independent samples.

## Reproduce

Install requirements-lock.txt and requirements-model-lock.txt with Python 3.12, then run
`python model_validation.py`. Source hash, package versions, fold dates, parameters and
every held-out prediction are saved alongside this report. Original course notebook and
presentation remain historical artifacts with their original scores.

Method reference: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
