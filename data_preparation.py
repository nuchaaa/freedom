"""
Подготовка данных для реального датасета ОГПО.

Ключевая особенность: данные на уровне ВОДИТЕЛЕЙ, но премия/выплаты — на уровне ПОЛИСА.
Необходимо агрегировать водителей в полис, дедуплицировать финансы.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from config import (
    RANDOM_SEED, COL_CONTRACT, COL_PREMIUM, COL_PREMIUM_WO_TERM,
    COL_CLAIM_AMOUNT, COL_IS_CLAIM, COL_CLAIM_CNT, COL_EXPERIENCE_YEAR,
    COL_BONUS_MALUS, COL_CAR_AGE, COL_ENGINE_VOLUME, COL_ENGINE_POWER,
    COL_REGION_ID, COL_VEHICLE_TYPE_ID, COL_IS_INDIVIDUAL, COL_IS_RESIDENCE,
    COL_AGE_EXP_ID, ALL_SCORE_COLUMNS, SCORE_GROUPS,
    WOE_FEATURES, WOE_REGULARIZATION,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА И ВАЛИДАЦИЯ
# ═══════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Загрузка CSV с базовой валидацией."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"[LOAD] Загружено: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    print(f"[LOAD] Уникальных полисов: {df[COL_CONTRACT].nunique():,}")
    print(f"[LOAD] Уникальных водителей: {df['driver_iin'].nunique():,}")
    print(f"[LOAD] Claim Rate (строки): {df[COL_IS_CLAIM].mean():.2%}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. АГРЕГАЦИЯ: ВОДИТЕЛИ → ПОЛИС
# ═══════════════════════════════════════════════════════════════════════

def aggregate_to_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные с уровня водителей на уровень полисов.

    Финансовые поля (premium, claim_amount) — берём ПЕРВОЕ значение (они одинаковые).
    Водительские поля — агрегируем min/max/mean.
    SCORE-признаки — агрегируем min/max/mean по водителям в полисе.
    """
    # Определяем доступные SCORE-колонки
    available_scores = [c for c in ALL_SCORE_COLUMNS if c in df.columns]

    # ── Финансовые и полисные (берём первое — они дублируются) ────────
    policy_cols = [COL_CONTRACT, COL_PREMIUM, COL_PREMIUM_WO_TERM,
                   COL_CLAIM_AMOUNT, COL_IS_CLAIM, COL_CLAIM_CNT,
                   COL_CAR_AGE, COL_ENGINE_VOLUME, COL_ENGINE_POWER,
                   COL_REGION_ID, COL_VEHICLE_TYPE_ID, COL_IS_INDIVIDUAL, COL_IS_RESIDENCE]
    policy_cols = [c for c in policy_cols if c in df.columns]
    policy_df = df.groupby(COL_CONTRACT)[policy_cols[1:]].first().reset_index()

    # ── Водительские агрегаты ────────────────────────────────────────
    driver_agg = {}

    # Кол-во водителей
    driver_agg[("driver_iin", "nunique")] = "n_drivers"

    # Стаж вождения
    if COL_EXPERIENCE_YEAR in df.columns:
        driver_agg[(COL_EXPERIENCE_YEAR, "min")] = "min_experience"
        driver_agg[(COL_EXPERIENCE_YEAR, "max")] = "max_experience"
        driver_agg[(COL_EXPERIENCE_YEAR, "mean")] = "mean_experience"

    # Бонус-Малус
    if COL_BONUS_MALUS in df.columns:
        driver_agg[(COL_BONUS_MALUS, "min")] = "min_bonus_malus"
        driver_agg[(COL_BONUS_MALUS, "max")] = "max_bonus_malus"
        driver_agg[(COL_BONUS_MALUS, "mean")] = "mean_bonus_malus"

    # Возраст-опыт ID
    if COL_AGE_EXP_ID in df.columns:
        driver_agg[(COL_AGE_EXP_ID, "min")] = "min_age_exp_id"
        driver_agg[(COL_AGE_EXP_ID, "max")] = "max_age_exp_id"

    # Формируем агрегацию
    agg_dict = {}
    rename_dict = {}
    for (col, func), new_name in driver_agg.items():
        if col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = []
            agg_dict[col].append(func)
            rename_dict[(col, func)] = new_name

    driver_df = df.groupby(COL_CONTRACT).agg(agg_dict)
    driver_df.columns = [rename_dict.get(c, f"{c[0]}_{c[1]}") for c in driver_df.columns]
    driver_df = driver_df.reset_index()

    # ── SCORE-агрегаты (min, max, mean по водителям) ─────────────────
    score_agg_dfs = []
    for group_name, cols in SCORE_GROUPS.items():
        group_cols = [c for c in cols if c in df.columns]
        if not group_cols:
            continue
        for func_name, func in [("min", "min"), ("max", "max"), ("mean", "mean")]:
            temp = df.groupby(COL_CONTRACT)[group_cols].agg(func)
            temp.columns = [f"{c}_{func_name}" for c in temp.columns]
            score_agg_dfs.append(temp)

    # Собираем всё вместе
    result = policy_df.merge(driver_df, on=COL_CONTRACT, how="left")
    for sdf in score_agg_dfs:
        result = result.merge(sdf, on=COL_CONTRACT, how="left")

    # Определяем расторжение: если premium_wo_term == 0 или сильно меньше premium
    if COL_PREMIUM_WO_TERM in result.columns and COL_PREMIUM in result.columns:
        result["is_canceled"] = (result[COL_PREMIUM_WO_TERM] <= 0).astype(int)
    else:
        result["is_canceled"] = 0

    print(f"[AGG] Агрегировано в {result.shape[0]:,} полисов × {result.shape[1]} столбцов")
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создание производных признаков на уровне полиса."""
    df = df.copy()

    # ── Бакеты для WoE ──────────────────────────────────────────────
    if "min_experience" in df.columns:
        df["experience_bucket"] = pd.cut(
            df["min_experience"].fillna(0),
            bins=[-1, 2, 5, 10, 20, 100],
            labels=["0-2", "3-5", "6-10", "11-20", "20+"],
        ).astype(str)

    if COL_ENGINE_POWER in df.columns:
        power_med = df[COL_ENGINE_POWER].median()
        df["engine_power_bucket"] = pd.cut(
            df[COL_ENGINE_POWER].fillna(power_med),
            bins=[0, 80, 120, 180, 300, 10000],
            labels=["low", "medium", "high", "very_high", "extreme"],
        ).astype(str)

    if "mean_bonus_malus" in df.columns:
        df["bonus_malus_bucket"] = pd.cut(
            df["mean_bonus_malus"].fillna(1),
            bins=[-np.inf, 0.7, 0.9, 1.0, 1.2, np.inf],
            labels=["very_good", "good", "neutral", "bad", "very_bad"],
        ).astype(str)

    # ── Производные признаки ─────────────────────────────────────────
    if "min_experience" in df.columns and COL_ENGINE_POWER in df.columns:
        df["power_per_exp"] = df[COL_ENGINE_POWER] / (df["min_experience"] + 1)

    if COL_ENGINE_POWER in df.columns and COL_ENGINE_VOLUME in df.columns:
        df["power_volume_ratio"] = df[COL_ENGINE_POWER] / (df[COL_ENGINE_VOLUME].replace(0, np.nan) + 1)

    if COL_PREMIUM in df.columns and COL_ENGINE_POWER in df.columns:
        df["premium_per_power"] = df[COL_PREMIUM] / (df[COL_ENGINE_POWER] + 1)

    if "n_drivers" in df.columns:
        df["multi_driver"] = (df["n_drivers"] > 1).astype(int)

    if COL_CAR_AGE in df.columns:
        df["old_car_flag"] = (df[COL_CAR_AGE] > 15).astype(int)

    # Region frequency encoding
    if COL_REGION_ID in df.columns:
        region_freq = df[COL_REGION_ID].value_counts(normalize=True).to_dict()
        df["region_freq"] = df[COL_REGION_ID].map(region_freq)

    # Vehicle type frequency encoding
    if COL_VEHICLE_TYPE_ID in df.columns:
        vtype_freq = df[COL_VEHICLE_TYPE_ID].value_counts(normalize=True).to_dict()
        df["vehicle_type_freq"] = df[COL_VEHICLE_TYPE_ID].map(vtype_freq)

    print(f"[FE] Feature engineering завершён. Признаков: {df.shape[1]}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. WoE ТРАНСФОРМАЦИЯ
# ═══════════════════════════════════════════════════════════════════════

class WoETransformer:
    """Weight of Evidence с Лапласовым сглаживанием."""

    def __init__(self, features=None, target_col=COL_IS_CLAIM, regularization=WOE_REGULARIZATION):
        self.features = features or WOE_FEATURES
        self.target_col = target_col
        self.reg = regularization
        self.woe_maps: Dict[str, Dict] = {}
        self.iv_values: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "WoETransformer":
        total_events = df[self.target_col].sum()
        total_non_events = len(df) - total_events

        for feat in self.features:
            if feat not in df.columns:
                print(f"[WoE] SKIP: {feat} не найден")
                continue
            grouped = df.groupby(feat)[self.target_col].agg(["sum", "count"])
            grouped.columns = ["events", "total"]
            grouped["non_events"] = grouped["total"] - grouped["events"]
            grouped["dist_events"] = (grouped["events"] + self.reg) / (total_events + self.reg * len(grouped))
            grouped["dist_non_events"] = (grouped["non_events"] + self.reg) / (total_non_events + self.reg * len(grouped))
            grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
            grouped["iv_component"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]
            self.woe_maps[feat] = grouped["woe"].to_dict()
            self.iv_values[feat] = grouped["iv_component"].sum()
            print(f"[WoE] {feat}: IV = {self.iv_values[feat]:.4f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feat in self.features:
            if feat in self.woe_maps:
                df[f"{feat}_woe"] = df[feat].map(self.woe_maps[feat]).fillna(0.0)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_iv_report(self) -> pd.DataFrame:
        rows = []
        for k, v in self.iv_values.items():
            power = "Not useful" if v < 0.02 else "Weak" if v < 0.1 else "Medium" if v < 0.3 else "Strong" if v < 0.5 else "Suspicious"
            rows.append({"feature": k, "IV": v, "predictive_power": power})
        return pd.DataFrame(rows).sort_values("IV", ascending=False)


# ═══════════════════════════════════════════════════════════════════════
# 5. ОПРЕДЕЛЕНИЕ ФИНАЛЬНЫХ ПРИЗНАКОВ
# ═══════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Динамически определяет список признаков для моделирования
    на основе доступных колонок в DataFrame.
    """
    feature_cols = []

    # Числовые базовые
    base_numeric = [
        COL_CAR_AGE, COL_ENGINE_VOLUME, COL_ENGINE_POWER,
        COL_REGION_ID, COL_VEHICLE_TYPE_ID, COL_IS_INDIVIDUAL, COL_IS_RESIDENCE,
        "n_drivers", "min_experience", "max_experience", "mean_experience",
        "min_bonus_malus", "max_bonus_malus", "mean_bonus_malus",
        "min_age_exp_id", "max_age_exp_id",
    ]

    # Производные
    derived = [
        "power_per_exp", "power_volume_ratio", "premium_per_power",
        "multi_driver", "old_car_flag", "region_freq", "vehicle_type_freq",
    ]

    # WoE
    woe_cols = [f"{f}_woe" for f in WOE_FEATURES]

    # SCORE агрегаты (все что содержат SCORE_ и заканчиваются на _min/_max/_mean)
    score_agg_cols = [c for c in df.columns if c.startswith("SCORE_") and
                      any(c.endswith(s) for s in ["_min", "_max", "_mean"])]

    for col in base_numeric + derived + woe_cols + score_agg_cols:
        if col in df.columns:
            feature_cols.append(col)

    print(f"[FEATURES] Отобрано {len(feature_cols)} признаков для моделирования")
    return feature_cols


# ═══════════════════════════════════════════════════════════════════════
# 6. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ (для тестирования)
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_real_format(n_policies=50_000, n_drivers=80_000, seed=RANDOM_SEED):
    """
    Генерирует синтетические данные в формате РЕАЛЬНОГО датасета
    (уровень водителей, с contract_number, SCORE-колонками и т.д.)
    """
    rng = np.random.default_rng(seed)

    # Генерируем полисы
    policy_ids = np.arange(1, n_policies + 1)
    driver_policy_ids = rng.choice(policy_ids, size=n_drivers, replace=True)

    # Водительские данные
    driver_age_raw = rng.integers(18, 75, size=n_drivers)
    experience = np.clip(driver_age_raw - 18 - rng.integers(0, 5, size=n_drivers), 0, 57)

    rows = {
        "unique_id": np.arange(1, n_drivers + 1),
        COL_CONTRACT: driver_policy_ids,
        "driver_iin": [f"DRV{i:08d}" for i in range(1, n_drivers + 1)],
        "insurer_iin": [f"INS{p:08d}" for p in driver_policy_ids],
        COL_EXPERIENCE_YEAR: experience,
        COL_AGE_EXP_ID: rng.integers(1, 10, size=n_drivers),
        COL_BONUS_MALUS: np.round(rng.uniform(0.5, 2.5, size=n_drivers), 2),
    }

    # SCORE-признаки (случайные для синтетики)
    for col in ALL_SCORE_COLUMNS[:20]:  # Берём первые 20 для скорости
        rows[col] = np.round(rng.normal(0, 1, size=n_drivers), 4)

    # Полисные данные (дублируются для всех водителей одного полиса)
    policy_data = {}
    policy_data["car_age"] = rng.integers(0, 25, size=n_policies)
    policy_data["engine_power"] = rng.choice([60, 80, 100, 120, 150, 200, 250], size=n_policies)
    policy_data["engine_volume"] = rng.choice([1.0, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0], size=n_policies)
    policy_data["region_id"] = rng.integers(1, 18, size=n_policies)
    policy_data["vehicle_type_id"] = rng.integers(1, 6, size=n_policies)
    policy_data["is_individual_person"] = rng.choice([0, 1], size=n_policies, p=[0.1, 0.9])
    policy_data["is_residence"] = rng.choice([0, 1], size=n_policies, p=[0.05, 0.95])

    base_premium = 15_000 + policy_data["engine_power"] * 80 + rng.normal(0, 2000, n_policies)
    premium = np.clip(base_premium, 5000, 150_000).astype(int)
    is_canceled = rng.binomial(1, 0.08, size=n_policies)
    premium_wo_term = np.where(is_canceled, 0, premium)

    claim_prob = 0.03 + (policy_data["engine_power"] / 300) * 0.05
    is_claim = rng.binomial(1, np.clip(claim_prob, 0.01, 0.15))
    severity = np.where(is_claim, np.clip(rng.lognormal(11.5, 1.2, n_policies), 10_000, 5_000_000), 0).astype(int)

    # Маппим полисные данные на уровень водителей
    policy_map_idx = driver_policy_ids - 1  # 0-indexed
    for col, vals in policy_data.items():
        rows[col] = vals[policy_map_idx]
    rows[COL_PREMIUM] = premium[policy_map_idx]
    rows[COL_PREMIUM_WO_TERM] = premium_wo_term[policy_map_idx]
    rows[COL_CLAIM_AMOUNT] = severity[policy_map_idx]
    rows[COL_IS_CLAIM] = is_claim[policy_map_idx]
    rows[COL_CLAIM_CNT] = is_claim[policy_map_idx]

    df = pd.DataFrame(rows)
    print(f"[SYNTH] Сгенерировано {n_drivers:,} строк (водители) для {n_policies:,} полисов")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 7. ПОЛНЫЙ PIPELINE ПОДГОТОВКИ
# ═══════════════════════════════════════════════════════════════════════

def prepare_dataset(path: Optional[str] = None) -> Tuple[pd.DataFrame, WoETransformer, list]:
    """
    Полный пайплайн: загрузка → агрегация → FE → WoE.

    Args:
        path: путь к CSV/parquet. Если None — генерирует синтетику.

    Returns:
        df: агрегированный DataFrame на уровне полисов
        woe: обученный WoE-трансформер
        feature_cols: список признаков для моделирования
    """
    if path:
        try:
            raw_df = load_data(path)
        except FileNotFoundError:
            print(f"[WARN] Файл '{path}' не найден! Генерируем синтетику...")
            raw_df = generate_synthetic_real_format()
    else:
        print("[WARN] Файл данных не указан — генерируем синтетику...")
        raw_df = generate_synthetic_real_format()

    df = aggregate_to_policy(raw_df)
    df = create_features(df)

    # WoE на доступных бакетах
    available_woe = [f for f in WOE_FEATURES if f in df.columns]
    woe = WoETransformer(features=available_woe)
    df = woe.fit_transform(df)

    feature_cols = get_feature_columns(df)

    print(f"\n[READY] Финальный датасет: {df.shape[0]:,} полисов × {df.shape[1]} столбцов")
    print(f"[READY] Claim Rate: {df[COL_IS_CLAIM].mean():.2%}")
    print(f"[READY] Ср. премия: {df[COL_PREMIUM].mean():,.0f} ₸")
    return df, woe, feature_cols
