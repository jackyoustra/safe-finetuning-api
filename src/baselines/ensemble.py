# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load blackbox model results
blackbox_results = pd.read_csv('results/prompt_summary.csv')

print(f"Blackbox results shape: {blackbox_results.shape}")
print("\nColumns in blackbox_results:")
print(blackbox_results.columns.tolist())

# Define test datasets (CMFT and benign)
cmft_datasets = [
    'SimpleRSACipher-p_17_q_23',
    'ASCIICipher-default',
    'WalnutSubstitutionCipher-seed_50',
    'WalnutSubstitutionCipher-seed_51'
]

benign_datasets = [
    'pure-dove',
    'oasst2',
    'long-protein'
]


# Combine all test datasets
test_datasets = cmft_datasets + benign_datasets

# Split data: everything not in test_datasets goes to train
test_df = blackbox_results[blackbox_results['dataset'].isin(test_datasets)].copy()
train_df = blackbox_results[~blackbox_results['dataset'].isin(test_datasets)].copy()

print("\nData split statistics:")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

print("\nTraining datasets:", train_df['dataset'].unique())
print("Testing datasets:", test_df['dataset'].unique())

# Prepare features and labels
evaluation_types = ['self-reflection', 'moderation', 'frontier']
for eval_type in evaluation_types:
    train_df[f'is_{eval_type.replace("-", "_")}'] = (train_df['evaluation_type'] == eval_type).astype(int)
    test_df[f'is_{eval_type.replace("-", "_")}'] = (test_df['evaluation_type'] == eval_type).astype(int)

# List of feature columns
feature_cols = [f'is_{eval_type.replace("-", "_")}' for eval_type in evaluation_types]

# Map 'decision' to binary labels
decision_mapping = {'SAFE': 0, 'UNSAFE': 1}
train_df['label'] = train_df['decision'].map(decision_mapping)
test_df['label'] = test_df['decision'].map(decision_mapping)

# Prepare feature matrices and label vectors
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_test = test_df[feature_cols].values
y_test = test_df['label'].values

print(f"\nNumber of features: {len(feature_cols)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\nTraining Logistic Regression...")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]

# Train XGBoost
print("Training XGBoost...")
xgb_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'n_estimators': 100,
    'tree_method': 'hist',
    'random_state': 42
}
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_scaled, y_train)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate models
def calculate_metrics(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    return {
        'AUROC': roc_auc_score(y_true, y_pred_probs) if len(np.unique(y_true)) > 1 else np.nan,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }

# Calculate and display metrics
metrics = {
    'Logistic Regression': calculate_metrics(y_test, logreg_probs),
    'XGBoost': calculate_metrics(y_test, xgb_probs)
}

metrics_df = pd.DataFrame(metrics).T
print("\nModel Performance Metrics:")
print(metrics_df)

# Plot ROC curves
plt.figure(figsize=(10, 6))
if len(np.unique(y_test)) > 1:
    # Logistic Regression ROC
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg_probs)
    roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
    plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
    
    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()

# Plot feature importance
xgb.plot_importance(xgb_model, importance_type='gain', xlabel='Gain', height=0.5)
plt.title('XGBoost Feature Importance')
plt.show()

# Save results
metrics_df.to_csv('ensemble_metrics_only_blackbox.csv', index=True)
print("\nMetrics saved to 'ensemble_metrics_only_blackbox.csv'")

# Save detailed predictions
results_df = test_df.copy()
results_df['logreg_prob'] = logreg_probs
results_df['xgb_prob'] = xgb_probs
results_df['logreg_pred'] = (logreg_probs >= 0.5).astype(int)
results_df['xgb_pred'] = (xgb_probs >= 0.5).astype(int)

results_df.to_csv('ensemble_detailed_results_only_blackbox.csv', index=False)
print("Detailed results saved to 'ensemble_detailed_results_only_blackbox.csv'")
