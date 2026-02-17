from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rcParams


def _set_plot_font() -> None:
    """
    Configure matplotlib font to support CJK labels on Windows/macOS/Linux.
    Falls back silently when no candidate font is found.
    """
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name] + list(rcParams.get("font.sans-serif", []))
            break
    # Avoid minus sign rendering as a square under some CJK fonts.
    rcParams["axes.unicode_minus"] = False


def plot_cumulative_return(cum_series: pd.Series, output_path: Path) -> None:
    """Save cumulative return line chart."""
    _set_plot_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.plot(cum_series.index, cum_series.values, label="Strategy")
    plt.title("Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_score_distribution(score_series: pd.Series, output_path: Path) -> None:
    """Save histogram of predicted probabilities."""
    _set_plot_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(score_series.dropna().values, bins=50)
    plt.title("Predicted Score Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, top_n: int = 30) -> None:
    """Save horizontal bar plot for top-N feature importance."""
    _set_plot_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = importance_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
