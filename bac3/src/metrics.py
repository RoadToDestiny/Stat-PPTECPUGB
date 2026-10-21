"""
Evaluation metrics and plotting utilities for chess blunder prediction.
Includes ROC-AUC, PR-AUC, Brier Score, and calibration curves.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def calculate_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes key probabilistic metrics required by the research methodology:
    - ROC-AUC
    - PR-AUC (Average Precision)
    - Brier Score
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
    }


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 10,
    model_name: str = "XGBoost"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plots and saves probability calibration curve.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"{model_name}")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (True Blunder Rate)")
    plt.title("Probability Calibration Curve")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    return prob_true, prob_pred


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Optional[str] = None,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Plots and saves feature importance bar chart.
    """
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=True)

    if len(df_imp) > top_n:
        df_imp = df_imp.tail(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(df_imp["feature"], df_imp["importance"], color="#1f77b4")
    plt.xlabel("Feature Importance (Gain / Weight)")
    plt.title(f"Top {len(df_imp)} Predictive Features for Chess Blunders")
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    return df_imp.sort_values(by="importance", ascending=False)
