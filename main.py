#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  ОГПО Loss Ratio Optimization Pipeline
  
  Поддерживает:
    - Реальный CSV/parquet (уровень водителей, агрегация в полисы)
    - Синтетику в формате реального датасета (для тестирования)
═══════════════════════════════════════════════════════════════
"""

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import TARGET_LOSS_RATIO, DATA_PATH
from data_preparation import prepare_dataset
from modeling import train_pipeline
from pricing_optimization import (
    compute_old_loss_ratios, optimize_margin_grid,
    optimize_margin_scipy, optimize_two_group_margins, generate_report,
)


def main(data_path=None):
    """
    Args:
        data_path: путь к CSV/parquet. None → синтетика.
    """
    path = data_path or (sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)

    # ── ЭТАП 1: Данные ──────────────────────────────────────────────
    print(f"\n{'█' * 70}\n  ЭТАП 1: ПОДГОТОВКА ДАННЫХ\n{'█' * 70}")
    df, woe, feature_cols = prepare_dataset(path)
    print("\n── WoE Information Value ───")
    print(woe.get_iv_report().to_string(index=False))

    # ── ЭТАП 2: Модели ──────────────────────────────────────────────
    print(f"\n{'█' * 70}\n  ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ\n{'█' * 70}")
    test_df, freq, sev, metrics = train_pipeline(df, feature_cols)

    old_lr = compute_old_loss_ratios(test_df)
    print(f"\n[BASELINE] LR на старых ценах: {old_lr['old_loss_ratio']:.2%}")

    # ── ЭТАП 3: Оптимизация ─────────────────────────────────────────
    print(f"\n{'█' * 70}\n  ЭТАП 3: ОПТИМИЗАЦИЯ\n{'█' * 70}")
    best_grid, grid_results = optimize_margin_grid(test_df)
    best_scipy = optimize_margin_scipy(test_df)
    best_m1, best_m2 = optimize_two_group_margins(test_df)

    # ── ЭТАП 4: Отчёт ──────────────────────────────────────────────
    print(f"\n{'█' * 70}\n  ЭТАП 4: ФИНАЛЬНЫЙ ОТЧЁТ\n{'█' * 70}")
    df_final = generate_report(test_df, best_margin=best_grid,
                               target_lr=TARGET_LOSS_RATIO, metrics=metrics)

    print(f"\n{'=' * 70}")
    print(f"  ✅ PIPELINE ЗАВЕРШЁН")
    print(f"  Grid margin: {best_grid:.4f} | Scipy margin: {best_scipy:.4f}")
    print(f"  Two-group: G1={best_m1:.4f}, G2={best_m2:.4f}")
    print(f"  GINI: {metrics['frequency']['test']['gini']:.4f}")
    print(f"{'=' * 70}")

    return df_final, freq, sev, metrics


if __name__ == "__main__":
    main()
