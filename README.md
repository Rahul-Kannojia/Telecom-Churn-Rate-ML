# Telecom-Churn-Rate-ML
## Project Overview
End-to-end churn classification pipeline on a synthetic telecom dataset — 
covering EDA, preprocessing, feature engineering, feature selection, 
5-model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM, 
Naive Bayes), and hyperparameter tuning.

## Key Finding: Diagnosing Weak Signal Over Chasing Metrics
Initial accuracy looked strong (~82%), but this was misleading — the dataset 
is ~80/20 imbalanced, meaning a model that always predicts "no churn" would 
already score ~80% accuracy while having zero predictive skill.

To validate this, I ran:
- **Correlation analysis**: features showed near-zero correlation with the 
  target and with each other
- **Information Value (IV) analysis**: flagged 2 features as strong 
  predictors by IV score alone
- **Train-set AUC cross-check**: even an unregularized, overfitting-permitted 
  Random Forest (train AUC = 1.00) collapsed to ~0.50 AUC on test — proving 
  the "signal" IV suggested didn't hold up under a stricter test

**Conclusion**: All 5 models converged on AUC ≈ 0.50-0.53 — consistent with 
no learnable relationship between the synthetic features and target, not a 
model or tuning failure. Prioritized diagnosing this correctly over 
over-tuning against noise.

## Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, scorecardpy
