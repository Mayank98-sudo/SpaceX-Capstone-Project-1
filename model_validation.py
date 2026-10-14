"""Reproducible historical holdout; no tuning or preprocessing on the test set."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
NUMERIC = ['FlightNumber', 'GridFins', 'Reused', 'Legs', 'Block']
CATEGORICAL = ['Orbit', 'LaunchSite']
FEATURES = NUMERIC + CATEGORICAL


def load_data(path=ROOT / 'dataset_part_2.csv'):
    frame = pd.read_csv(path)
    frame['Date'] = pd.to_datetime(frame['Date'], errors='raise')
    if frame.empty or frame['FlightNumber'].duplicated().any():
        raise ValueError('Empty data or duplicate flight identifiers')
    if frame[FEATURES + ['Class', 'Date']].isna().any().any():
        raise ValueError('Unexpected missing values in the selected source columns')
    if not set(frame['Class'].unique()).issubset({0, 1}):
        raise ValueError('Landing label must be binary')
    return frame.sort_values(['Date', 'FlightNumber']).reset_index(drop=True)


def chronological_split(frame, test_fraction=0.2):
    """Keep missions on the same day together; never split one day across sets."""
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    ordered = frame.sort_values(['Date', 'FlightNumber']).reset_index(drop=True)
    boundary = ordered.iloc[int(len(ordered) * (1 - test_fraction))]['Date']
    train = ordered[ordered['Date'] < boundary].copy()
    test = ordered[ordered['Date'] >= boundary].copy()
    if train.empty or test.empty or train['Class'].nunique() != 2:
        raise ValueError('Need nonempty holdout and two training classes')
    return train, test


def expanding_date_folds(frame):
    """Three validation windows with strictly earlier training dates."""
    dates = np.array_split(np.sort(frame['Date'].unique()), 4)
    folds = []
    for window in dates[1:]:
        train = np.flatnonzero((frame['Date'] < window[0]).to_numpy())
        validation = np.flatnonzero(frame['Date'].isin(window).to_numpy())
        if len(train) < 7 or frame.iloc[train]['Class'].nunique() != 2:
            raise ValueError('Insufficient early history for temporal tuning')
        folds.append((train, validation))
    return folds


def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])


def candidates():
    return {
        'Majority baseline': (DummyClassifier(strategy='most_frequent'), {}),
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=42),
                                {'model__C': [0.01, 0.1, 1, 10]}),
        'SVM': (SVC(random_state=42), {'model__C': [0.1, 1, 10],
                                     'model__kernel': ['linear', 'rbf']}),
        'Decision Tree': (DecisionTreeClassifier(random_state=42),
                          {'model__max_depth': [2, 3, 5], 'model__min_samples_leaf': [1, 3, 5]}),
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7],
                                         'model__weights': ['uniform', 'distance']}),
    }


def score_predictions(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(actual, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(actual, predicted)),
        'precision': float(precision_score(actual, predicted, zero_division=0)),
        'recall': float(recall_score(actual, predicted, zero_division=0)),
        'f1': float(f1_score(actual, predicted, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def run(output=ROOT / 'reports' / 'validation'):
    frame = load_data()
    train, test = chronological_split(frame)
    train = train.reset_index(drop=True)
    folds = expanding_date_folds(train)
    results, searches = [], {}
    # Complete every tuning decision before examining any holdout predictions.
    for name, (estimator, grid) in candidates().items():
        pipe = Pipeline([('preprocess', preprocessing()), ('model', estimator)])
        search = GridSearchCV(pipe, grid, scoring='balanced_accuracy', cv=folds,
                              n_jobs=1, error_score='raise', refit=True)
        search.fit(train[FEATURES], train['Class'])
        searches[name] = search
    selected = max(searches, key=lambda name: searches[name].best_score_)
    predictions = test[['FlightNumber', 'Date', 'Class']].copy()
    for name, search in searches.items():
        prediction = search.predict(test[FEATURES])
        predictions[name] = prediction
        results.append({'model': name, 'cv_balanced_accuracy': float(search.best_score_),
                        'parameters': search.best_params_, **score_predictions(test['Class'], prediction)})
    fold_report = [{'train_count': len(a), 'validation_count': len(b),
                    'train_end': train.iloc[a]['Date'].max().date().isoformat(),
                    'validation_start': train.iloc[b]['Date'].min().date().isoformat(),
                    'validation_end': train.iloc[b]['Date'].max().date().isoformat(),
                    'validation_success_count': int(train.iloc[b]['Class'].sum())}
                   for a, b in folds]
    metadata = {
        'source_sha256': hashlib.sha256((ROOT / 'dataset_part_2.csv').read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
        'features': FEATURES, 'selected_by_training_cv': selected,
        'rows': len(frame), 'training_rows': len(train), 'test_rows': len(test),
        'train_start': train['Date'].min().date().isoformat(),
        'train_end': train['Date'].max().date().isoformat(),
        'test_start': test['Date'].min().date().isoformat(),
        'test_end': test['Date'].max().date().isoformat(),
        'test_successes': int(test['Class'].sum()), 'folds': fold_report, 'results': results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    predictions.to_csv(output / 'holdout_predictions.csv', index=False)
    table = ['| Model | CV balanced accuracy | Holdout accuracy | Balanced accuracy | Precision | Recall | F1 | TN / FP / FN / TP |',
             '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |']
    for row in results:
        table.append('| ' + row['model'] + ' | ' + ' | '.join(f"{row[k]:.1%}" for k in
                     ['cv_balanced_accuracy', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']) +
                     f" | {row['tn']} / {row['fp']} / {row['fn']} / {row['tp']} |")
    text = f'''# Falcon 9: chronological validation

An independently rerun educational analysis of {len(frame)} historical missions.
Train: **{metadata['train_start']} to {metadata['train_end']} ({len(train)} missions)**.
Holdout: **{metadata['test_start']} to {metadata['test_end']} ({len(test)} missions)**,
with **{metadata['test_successes']} successful and {len(test) - metadata['test_successes']} unsuccessful labels**.

Selected using only training cross-validation: **{selected}**. All candidates' holdout
metrics below are descriptive; do not select a new winner using this test set.

''' + '\n'.join(table) + '''

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
'''
    (output / 'REPORT.md').write_text(text, encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(results))
    ax.bar(positions - .18, [r['accuracy'] for r in results], .36, label='Holdout accuracy')
    ax.bar(positions + .18, [r['balanced_accuracy'] for r in results], .36, label='Holdout balanced accuracy')
    ax.set_xticks(positions, [r['model'] for r in results], rotation=15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title(f'Falcon 9: {len(test)} later missions; {metadata["test_successes"]} success labels')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'holdout-comparison.svg', metadata={'Date': None})
    plt.close(fig)
    print(text)
    return metadata


if __name__ == '__main__':
    run()
