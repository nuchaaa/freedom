"""
Ценообразование и оптимизация Loss Ratio.
Адаптировано: LR = Выплаты / Премии за вычетом расторжений.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional

from config import (
    TARGET_LOSS_RATIO, MIN_PRICE_MULTIPLIER, MAX_PRICE_MULTIPLIER,
    OPTIM_MARGIN_GRID_POINTS, OPTIM_MARGIN_RANGE,
    COL_PREMIUM, COL_PREMIUM_WO_TERM, COL_CLAIM_AMOUNT,
)


def compute_new_premium(df, margin_coeff=1.0, target_lr=TARGET_LOSS_RATIO,
                        min_mult=MIN_PRICE_MULTIPLIER, max_mult=MAX_PRICE_MULTIPLIER):
    """
    new_premium = pure_premium / target_lr * margin_coeff
    Клиппинг: [old_premium * min_mult, old_premium * max_mult]
    """
    df = df.copy()
    old_p = df[COL_PREMIUM]
    raw = (df["pure_premium"] / target_lr) * margin_coeff
    df["new_premium"] = np.clip(raw, old_p * min_mult, old_p * max_mult)
    df["price_change_ratio"] = df["new_premium"] / old_p.replace(0, 1)
    df["group"] = np.where(df["price_change_ratio"] <= 1.0, "group_1", "group_2")

    # new_premium_wo_term: пропорционально масштабируем
    if COL_PREMIUM_WO_TERM in df.columns:
        scale = df["new_premium"] / old_p.replace(0, 1)
        df["new_premium_wo_term"] = df[COL_PREMIUM_WO_TERM] * scale
    else:
        df["new_premium_wo_term"] = np.where(df["is_canceled"] == 0, df["new_premium"], 0)
    return df


def compute_loss_ratios(df):
    """LR = Σ claim_amount / Σ new_premium_wo_term (по нерасторгнутым)."""
    results = {}
    for label, subset in [("total", df), ("group_1", df[df["group"] == "group_1"]),
                          ("group_2", df[df["group"] == "group_2"])]:
        if len(subset) == 0:
            results[label] = {"loss_ratio": np.nan, "n_policies": 0,
                              "total_claims": 0, "total_premiums": 0, "share_of_portfolio": 0}
            continue
        total_premiums = subset["new_premium_wo_term"].sum()
        total_claims = subset[COL_CLAIM_AMOUNT].sum()
        lr = total_claims / total_premiums if total_premiums > 0 else np.inf
        results[label] = {
            "loss_ratio": lr, "n_policies": len(subset),
            "total_claims": total_claims, "total_premiums": total_premiums,
            "share_of_portfolio": len(subset) / len(df),
        }
    return results


def compute_old_loss_ratios(df):
    """Baseline LR на старых премиях."""
    col = COL_PREMIUM_WO_TERM if COL_PREMIUM_WO_TERM in df.columns else COL_PREMIUM
    total_premiums = df[col].sum()
    total_claims = df[COL_CLAIM_AMOUNT].sum()
    lr = total_claims / total_premiums if total_premiums > 0 else np.inf
    return {"old_loss_ratio": lr, "total_claims": total_claims, "total_old_premiums": total_premiums}


def _loss_function(margin_coeff, df, target_lr):
    df_priced = compute_new_premium(df, margin_coeff=margin_coeff, target_lr=target_lr)
    lr = compute_loss_ratios(df_priced)
    lr_t = lr["total"]["loss_ratio"]
    lr_g1 = lr["group_1"]["loss_ratio"] if not np.isnan(lr["group_1"]["loss_ratio"]) else target_lr
    lr_g2 = lr["group_2"]["loss_ratio"] if not np.isnan(lr["group_2"]["loss_ratio"]) else target_lr
    share_g1 = lr["group_1"]["share_of_portfolio"]
    return (3.0 * (lr_t - target_lr)**2 + 1.5 * (lr_g1 - target_lr)**2
            + 1.5 * (lr_g2 - target_lr)**2 - 0.5 * share_g1)


def optimize_margin_grid(df, target_lr=TARGET_LOSS_RATIO,
                         n_points=OPTIM_MARGIN_GRID_POINTS,
                         margin_range=OPTIM_MARGIN_RANGE):
    """Grid-search по маржинальному коэффициенту."""
    print(f"\n{'=' * 70}\n  GRID SEARCH OPTIMIZATION\n{'=' * 70}")
    margins = np.linspace(*margin_range, n_points)
    results = []
    for m in margins:
        df_p = compute_new_premium(df, margin_coeff=m, target_lr=target_lr)
        lr = compute_loss_ratios(df_p)
        results.append({
            "margin_coeff": m,
            "lr_total": lr["total"]["loss_ratio"],
            "lr_group1": lr["group_1"]["loss_ratio"],
            "lr_group2": lr["group_2"]["loss_ratio"],
            "share_group1": lr["group_1"]["share_of_portfolio"],
            "loss": _loss_function(m, df, target_lr),
        })
    rdf = pd.DataFrame(results)
    best_idx = rdf["loss"].idxmin()
    best = rdf.loc[best_idx]
    print(f"[OPT] Best margin: {best['margin_coeff']:.4f}")
    print(f"[OPT] LR total: {best['lr_total']:.2%} | G1: {best['lr_group1']:.2%} | G2: {best['lr_group2']:.2%}")
    print(f"[OPT] Share G1: {best['share_group1']:.2%}")
    return best["margin_coeff"], rdf


def optimize_margin_scipy(df, target_lr=TARGET_LOSS_RATIO, margin_range=OPTIM_MARGIN_RANGE):
    """SciPy Brent optimization."""
    result = minimize_scalar(_loss_function, bounds=margin_range, method="bounded",
                             args=(df, target_lr), options={"xatol": 1e-4})
    print(f"[OPT-Scipy] Best margin: {result.x:.4f} | Loss: {result.fun:.6f}")
    return result.x


def optimize_two_group_margins(df, target_lr=TARGET_LOSS_RATIO, n_points=60):
    """Двухфакторная оптимизация: разные маржи для Группы 1 и 2."""
    print(f"\n{'=' * 70}\n  TWO-GROUP OPTIMIZATION\n{'=' * 70}")
    best_loss, best_m1, best_m2 = np.inf, 1.0, 1.0
    m1s = np.linspace(0.5, 2.0, n_points)
    m2s = np.linspace(0.8, 2.5, n_points)

    for m1 in m1s:
        for m2 in m2s:
            df_base = compute_new_premium(df, margin_coeff=1.0, target_lr=target_lr)
            g1_mask = df_base["group"] == "group_1"
            g2_mask = df_base["group"] == "group_2"
            df_p = df_base.copy()
            if g1_mask.any():
                t = compute_new_premium(df[g1_mask], margin_coeff=m1, target_lr=target_lr)
                df_p.loc[g1_mask, "new_premium"] = t["new_premium"].values
                df_p.loc[g1_mask, "new_premium_wo_term"] = t["new_premium_wo_term"].values
                df_p.loc[g1_mask, "price_change_ratio"] = t["price_change_ratio"].values
            if g2_mask.any():
                t = compute_new_premium(df[g2_mask], margin_coeff=m2, target_lr=target_lr)
                df_p.loc[g2_mask, "new_premium"] = t["new_premium"].values
                df_p.loc[g2_mask, "new_premium_wo_term"] = t["new_premium_wo_term"].values
                df_p.loc[g2_mask, "price_change_ratio"] = t["price_change_ratio"].values
            lr = compute_loss_ratios(df_p)
            lr_t = lr["total"]["loss_ratio"]
            lr_g1 = lr["group_1"]["loss_ratio"] if not np.isnan(lr["group_1"]["loss_ratio"]) else target_lr
            lr_g2 = lr["group_2"]["loss_ratio"] if not np.isnan(lr["group_2"]["loss_ratio"]) else target_lr
            loss = 3*(lr_t-target_lr)**2 + 2*(lr_g1-target_lr)**2 + 2*(lr_g2-target_lr)**2 - 0.3*lr["group_1"]["share_of_portfolio"]
            if loss < best_loss:
                best_loss, best_m1, best_m2 = loss, m1, m2

    # Показать результат
    df_base = compute_new_premium(df, margin_coeff=1.0, target_lr=target_lr)
    g1 = df_base["group"] == "group_1"
    g2 = df_base["group"] == "group_2"
    df_f = df_base.copy()
    if g1.any():
        t = compute_new_premium(df[g1], margin_coeff=best_m1, target_lr=target_lr)
        df_f.loc[g1, ["new_premium","new_premium_wo_term","price_change_ratio"]] = t[["new_premium","new_premium_wo_term","price_change_ratio"]].values
    if g2.any():
        t = compute_new_premium(df[g2], margin_coeff=best_m2, target_lr=target_lr)
        df_f.loc[g2, ["new_premium","new_premium_wo_term","price_change_ratio"]] = t[["new_premium","new_premium_wo_term","price_change_ratio"]].values
    lr_f = compute_loss_ratios(df_f)
    print(f"[OPT-2G] G1={best_m1:.4f}, G2={best_m2:.4f}")
    print(f"[OPT-2G] LR total: {lr_f['total']['loss_ratio']:.2%} | G1: {lr_f['group_1']['loss_ratio']:.2%} | G2: {lr_f['group_2']['loss_ratio']:.2%}")
    print(f"[OPT-2G] Share G1: {lr_f['group_1']['share_of_portfolio']:.2%}")
    return best_m1, best_m2


def generate_report(df, best_margin, target_lr=TARGET_LOSS_RATIO, metrics=None):
    """Итоговый отчёт: LR до/после, статистика цен, метрики моделей."""
    print(f"\n{'=' * 70}\n  ИТОГОВЫЙ ОТЧЁТ\n{'=' * 70}")

    old_lr = compute_old_loss_ratios(df)
    print(f"\n  BASELINE: LR = {old_lr['old_loss_ratio']:.2%}")
    print(f"  Выплаты: {old_lr['total_claims']:>15,.0f} ₸ | Премии: {old_lr['total_old_premiums']:>15,.0f} ₸")

    df_f = compute_new_premium(df, margin_coeff=best_margin, target_lr=target_lr)
    lr_new = compute_loss_ratios(df_f)

    print(f"\n  ПОСЛЕ ОПТИМИЗАЦИИ (margin = {best_margin:.4f}):")
    for label, name in [("total","ПОРТФЕЛЬ"),("group_1","ГРУППА 1 (↓/=)"),("group_2","ГРУППА 2 (↑)")]:
        info = lr_new[label]
        print(f"    {name}: {info['n_policies']:,} полисов ({info['share_of_portfolio']:.1%}) | LR = {info['loss_ratio']:.2%}")

    ratio = df_f["price_change_ratio"]
    print(f"\n  Изменение цен: mean={ratio.mean():.3f} | median={ratio.median():.3f} | min={ratio.min():.3f} | max={ratio.max():.3f}")
    print(f"  Доля ↓/=: {(ratio<=1).mean():.1%} | Доля ↑: {(ratio>1).mean():.1%}")

    if metrics:
        freq = metrics.get("frequency",{}).get("test",{})
        sev = metrics.get("severity",{})
        if freq: print(f"\n  Frequency: AUC={freq.get('roc_auc','?'):.4f} | GINI={freq.get('gini','?'):.4f}")
        if sev: print(f"  Severity: RMSE={sev.get('rmse','?'):,.0f} ₸")

    return df_f
