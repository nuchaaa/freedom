"""
Визуализация результатов оптимизации ОГПО.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def setup_style():
    """Настройка стиля графиков."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (14, 8),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_loss_ratio_comparison(
    old_lr: float,
    new_lr_total: float,
    new_lr_g1: float,
    new_lr_g2: float,
    target_lr: float = 0.70,
    save_path: Optional[str] = None,
):
    """Сравнение Loss Ratio до и после оптимизации."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Baseline\n(старые цены)", "После оптимизации\n(весь портфель)",
                  "Группа 1\n(цена ↓/=)", "Группа 2\n(цена ↑)"]
    values = [old_lr, new_lr_total, new_lr_g1, new_lr_g2]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

    bars = ax.bar(categories, values, color=colors, width=0.6, edgecolor="white", linewidth=2)

    # Целевая линия
    ax.axhline(y=target_lr, color="#2c3e50", linestyle="--", linewidth=2,
               label=f"Целевой LR = {target_lr:.0%}")

    # Значения над столбцами
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_ylabel("Loss Ratio")
    ax.set_title("Оптимизация убыточности ОГПО: Loss Ratio до и после", fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[VIZ] Сохранено: {save_path}")
    plt.show()


def plot_price_change_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Распределение изменений цен."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма
    ax = axes[0]
    ratio = df["price_change_ratio"]
    ax.hist(ratio, bins=80, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(x=1.0, color="#e74c3c", linestyle="--", linewidth=2, label="Без изменений (1.0)")
    ax.axvline(x=ratio.median(), color="#2ecc71", linestyle="-", linewidth=2,
               label=f"Медиана ({ratio.median():.2f})")
    ax.set_xlabel("Коэффициент изменения цены")
    ax.set_ylabel("Кол-во полисов")
    ax.set_title("Распределение изменений цен")
    ax.legend()

    # Boxplot по группам
    ax = axes[1]
    g1 = df[df["group"] == "group_1"]["price_change_ratio"]
    g2 = df[df["group"] == "group_2"]["price_change_ratio"]
    bp = ax.boxplot([g1, g2], labels=["Группа 1\n(↓/=)", "Группа 2\n(↑)"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor("#f39c12")
    ax.set_ylabel("Коэффициент изменения цены")
    ax.set_title("Изменение цен по группам")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_optimization_landscape(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """График ландшафта оптимизации margin_coeff."""
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss Ratios vs margin
    ax = axes[0]
    ax.plot(results_df["margin_coeff"], results_df["lr_total"], label="LR Total", linewidth=2)
    ax.plot(results_df["margin_coeff"], results_df["lr_group1"], label="LR Группа 1", linewidth=2)
    ax.plot(results_df["margin_coeff"], results_df["lr_group2"], label="LR Группа 2", linewidth=2)
    ax.axhline(y=0.70, color="red", linestyle="--", alpha=0.7, label="Target 70%")
    ax.set_ylabel("Loss Ratio")
    ax.set_title("Loss Ratio vs Маржинальный коэффициент")
    ax.legend()

    # Share of Group 1
    ax = axes[1]
    ax.fill_between(results_df["margin_coeff"], results_df["share_group1"],
                    alpha=0.3, color="#3498db")
    ax.plot(results_df["margin_coeff"], results_df["share_group1"],
            color="#3498db", linewidth=2, label="Доля Группы 1")
    ax.set_xlabel("Маржинальный коэффициент")
    ax.set_ylabel("Доля Группы 1")
    ax.set_title("Доля полисов со снижением/без изменения цены")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pure_premium_vs_actual(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Scatter: Pure Premium vs Old Premium, цвет = is_claim."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    claims = df[df["is_claim"] == 1]
    no_claims = df[df["is_claim"] == 0].sample(min(3000, len(df[df["is_claim"] == 0])))

    ax.scatter(no_claims["old_premium"], no_claims["pure_premium"],
               alpha=0.2, s=10, color="#3498db", label="Без ДТП")
    ax.scatter(claims["old_premium"], claims["pure_premium"],
               alpha=0.6, s=30, color="#e74c3c", label="С ДТП")

    # Диагональ
    lim = max(df["old_premium"].max(), df["pure_premium"].max()) * 0.8
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="x = y")

    ax.set_xlabel("Старая Премия (₸)")
    ax.set_ylabel("Pure Premium (₸)")
    ax.set_title("Pure Premium vs Старая Премия")
    ax.legend()
    ax.set_xlim(0, df["old_premium"].quantile(0.99))
    ax.set_ylim(0, df["pure_premium"].quantile(0.99))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
