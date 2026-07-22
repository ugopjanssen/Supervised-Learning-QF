# Supervised Learning in Return Expectations

Logistic regression with L1 (Lasso) and L2 (Ridge) regularisation applied to two short term return prediction tasks: next day direction for QQQ, and one month yield change direction for the 10 Year Treasury (^TNX).

## Overview

The notebook covers two independent classification tasks:

- **Group A — QQQ:** binary classification of next day price direction (up/down), using daily OHLCV data.
- **Group B — ^TNX:** three class classification of one month yield change (down/flat/up), based on tercile buckets.

Features are engineered from raw OHLCV data (O-C, H-L, sign of return, lagged returns, momentum, SMA, EMA), and both tasks are evaluated with confusion matrices, ROC-AUC, and precision/recall.

Short term returns in efficient markets are not expected to be strongly predictable. The point of this project is to demonstrate correct methodology and honest evaluation, not to claim a practical trading edge.

## Structure

1. Data acquisition and initial review (QQQ & ^TNX)
2. Feature construction for both groups
3. The logistic classifier and its formulation, including the mathematical basis of Ridge and Lasso, and a bias/variance comparison between the two
4. Model selection and evaluation: scaling comparison (min max vs. robust), collinearity and feature selection via VIF, sigmoid analysis, and the confusion matrix / ROC-AUC / precision-recall metrics
5. Mathematical bases of supervised learning and conclusion
6. Appendix: a 3D geometric visualisation of why Lasso's diamond shaped constraint region produces sparse solutions, while Ridge's circular region does not

## Results

Both models land close to their respective random baselines, consistent with the efficient markets framing above:

- **QQQ (binary):** test accuracy of around 54.8%, versus a roughly balanced 50% baseline.
- **^TNX (three class):** test accuracy of around 34.7%, versus a roughly balanced 33% baseline.

Lasso and Ridge show the expected qualitative difference in their fitted coefficients: Lasso zeroes out several of the noisier lagged return features outright, while Ridge shrinks all coefficients toward zero without eliminating any of them.

## Requirements

`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `yfinance`, `scipy`

## Inspiration

This project was inspired by the CQF Exam 3 brief (January 2021 Cohort, "Supervised Learning in Return Expectations"), which frames the same two group binary/multinomial classification setup, the Table 1 feature list, and the Ridge/Lasso and evaluation requirements this notebook builds on. It isn't a submission for that exam and doesn't follow the brief to the letter, but it's where the idea and structure originally came from.
