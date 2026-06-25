# Titanic Survival Prediction 

A first end-to-end machine-learning project: predicting which Titanic passengers
survived, using a logistic-regression classifier built in Python.

> **Independent learning project** — built to practice the full ML workflow from
> raw data to an evaluated model.

## Result

| Metric | Value |
|---|---|
| Model | Logistic Regression (scikit-learn) |
| **Accuracy on unseen test data** | **~81%** |
| Majority-class baseline | ~58% |
| **Lift over baseline** | **+23 points** |

The 23-point lift over the "always guess the most common outcome" baseline shows
the model learned real signal (sex, passenger class, and age were the strongest
drivers of survival) rather than parroting the majority class.

## What this project demonstrates

- **Data cleaning** — handled missing values (median imputation for `age`,
  mode for `embarked`; dropped `deck` at ~77% missing)
- **Feature selection** — removed duplicate/leakage columns (e.g. `alive`, which
  is the target in disguise) and redundant derived columns
- **Feature engineering** — one-hot encoded categorical columns (`sex`,
  `embarked`) while correctly leaving continuous columns (`age`, `fare`) as-is
- **Train/test split** — held out 20% of data to evaluate honestly on unseen rows
- **Modeling & evaluation** — trained logistic regression and benchmarked
  accuracy against a baseline

## Tech stack

Python · pandas · scikit-learn · seaborn · matplotlib · Jupyter

## How to run

```bash
# create the environment
uv venv
uv pip install pandas scikit-learn jupyter matplotlib seaborn

# launch the notebook
.venv/bin/jupyter lab
```

Then open `titanic_survival.ipynb` and run: **Kernel → Restart Kernel and Run All Cells**.

## Data

The classic Titanic dataset (891 passengers), loaded directly from seaborn
(`seaborn.load_dataset('titanic')`) — public sample data, no download required.

## What I'd do next

- Try additional models (Random Forest, Gradient Boosting) and compare
- Add cross-validation for a more robust accuracy estimate
- Engineer new features (e.g. family size = sibsp + parch, title from name)
- Look beyond accuracy at precision/recall and a confusion matrix
