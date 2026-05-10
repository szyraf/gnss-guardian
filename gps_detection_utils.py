"""
GNSS Guardian - Generic ML utilities shared between UAV and AV notebooks.

Domain-specific code (data generation, feature engineering, physics rules)
lives inside each notebook so the analysis tells a self-contained story.
This file holds only what is genuinely reusable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def train_rf(df, features, target, test_size=0.3, seed=42):
    """Train RandomForest with stratified split. Returns model + test set."""
    X = df[features].fillna(0)
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        random_state=seed, n_jobs=-1, class_weight='balanced'
    )
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


def evaluate(model, X_te, y_te, label="Model"):
    """Compute and print key metrics. Returns dict for downstream use."""
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    m = {
        'accuracy':  accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall':    recall_score(y_te, y_pred, zero_division=0),
        'f1':        f1_score(y_te, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_te, y_proba),
    }
    print(f"\n{label}")
    print("-" * len(label))
    for k, v in m.items():
        print(f"  {k:<10} {v:.4f}")
    return m, y_pred, y_proba


def plot_results(y_te, y_pred, importance_series, class_names, title=""):
    """Confusion matrix + feature importance side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    cm = confusion_matrix(y_te, y_pred)
    im = ax[0].imshow(cm, cmap='Blues')
    ax[0].set_title(f'{title} — Confusion matrix')
    ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('Actual')
    ax[0].set_xticks([0, 1], class_names); ax[0].set_yticks([0, 1], class_names)
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cm[i, j], ha='center', va='center',
                       color='white' if cm[i, j] > thr else 'black')
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    top = importance_series.sort_values(ascending=True).tail(10)
    ax[1].barh(top.index, top.values, color='steelblue')
    ax[1].set_title(f'{title} — Top features (Gini importance)')
    ax[1].set_xlabel('Importance')
    plt.tight_layout()
    plt.show()


def compare_methods(ml_metrics, rule_acc, n_samples, attack_rate, domain):
    """Pretty-print ML vs rules comparison."""
    bar = "=" * 56
    print(f"\n{bar}\n{domain} — ML vs Physics-rules baseline\n{bar}")
    print(f"Samples: {n_samples:,}   |   Attack rate: {attack_rate:.1%}")
    print(f"{'':<22}{'ML (RF)':<12}{'Rules':<12}")
    print(f"{'Accuracy':<22}{ml_metrics['accuracy']:<12.4f}{rule_acc:<12.4f}")
    delta = ml_metrics['accuracy'] - rule_acc
    print(f"{'ML lift over rules':<22}{delta:+.3f}")
    print(bar)
