"""
Конфигурация проекта: гиперпараметры, бизнес-ограничения, пути.

Адаптировано под реальный датасет ОГПО (уровень водителей).
Поддерживает два бэкенда: LightGBM (production) / Sklearn (fallback).
"""

import importlib

# ─────────────────────────── Авто-определение бэкенда ─────────────────
def _detect_backend() -> str:
    try:
        importlib.import_module("lightgbm")
        return "lightgbm"
    except Exception:
        return "sklearn"

MODEL_BACKEND = _detect_backend()

# ─────────────────────────── Путь к данным ────────────────────────────
DATA_PATH = "final_dataset.csv"  # Путь к CSV-файлу (можно переопределить в main.py)

# ─────────────────────────── Бизнес-правила ───────────────────────────
TARGET_LOSS_RATIO = 0.70          # Целевой Loss Ratio = 70%
MIN_PRICE_MULTIPLIER = 0.0       # Снижение цены максимум до 0 (−100%)
MAX_PRICE_MULTIPLIER = 3.0       # Повышение цены максимум в 3 раза

# ─────────────────────────── Данные ───────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.25
N_SYNTHETIC_POLICIES = 50_000    # Для синтетического режима
N_SYNTHETIC_DRIVERS = 80_000

# ─────────────────────────── Колонки реального датасета ────────────────
# Идентификаторы
COL_UNIQUE_ID = "unique_id"
COL_CONTRACT = "contract_number"

# Финансовые (на уровне полиса — одинаковые для всех водителей)
COL_PREMIUM = "premium"
COL_PREMIUM_WO_TERM = "premium_wo_term"  # Премия за вычетом расторжений
COL_CLAIM_AMOUNT = "claim_amount"
COL_CLAIM_CNT = "claim_cnt"
COL_IS_CLAIM = "is_claim"

# Водительские
COL_INSURER_IIN = "insurer_iin"
COL_DRIVER_IIN = "driver_iin"
COL_EXPERIENCE_YEAR = "experience_year"
COL_AGE_EXP_ID = "age_experience_id"
COL_AGE_EXP_NAME = "age_experience_name"
COL_BONUS_MALUS = "bonus_malus"

# Полис / ТС
COL_OPERATION_DATE = "operation_date"
COL_REGION_ID = "region_id"
COL_REGION_NAME = "region_name"
COL_VEHICLE_TYPE_ID = "vehicle_type_id"
COL_VEHICLE_TYPE_NAME = "vehicle_type_name"
COL_CAR_AGE = "car_age"
COL_CAR_YEAR = "car_year"
COL_ENGINE_VOLUME = "engine_volume"
COL_ENGINE_POWER = "engine_power"
COL_MODEL = "model"
COL_MARK = "mark"
COL_CAR_NUMBER = "car_number"
COL_OWNERKATO = "ownerkato"
COL_OWNERKATO_SHORT = "ownerkato_short"

# Флаги
COL_IS_INDIVIDUAL = "is_individual_person"
COL_IS_RESIDENCE = "is_residence"

# ─────────────────────────── SCORE-признаки (предикторы) ──────────────
SCORE_GROUPS = {
    "SCORE_1": [f"SCORE_1_{i}" for i in range(1, 11)],
    "SCORE_2": [f"SCORE_2_{i}" for i in range(1, 4)],
    "SCORE_3": [f"SCORE_3_{i}" for i in range(1, 11)],
    "SCORE_4": [f"SCORE_4_{i}" for i in range(1, 31)],
    "SCORE_5": [f"SCORE_5_{i}" for i in range(1, 13)],
    "SCORE_6": [f"SCORE_6_{i}" for i in range(1, 4)],
    "SCORE_7": [f"SCORE_7_{i}" for i in range(1, 4)],
    "SCORE_8": [f"SCORE_8_{i}" for i in range(1, 4)],
    "SCORE_9": [f"SCORE_9_{i}" for i in range(1, 25)],
    "SCORE_10": [f"SCORE_10_{i}" for i in range(1, 7)],
    "SCORE_11": [f"SCORE_11_{i}" for i in range(1, 15)],
    "SCORE_12": [f"SCORE_12_{i}" for i in range(1, 7)] + ["SCORE_12_9_1", "SCORE_12_7", "SCORE_12_8", "SCORE_12_10"],
}

ALL_SCORE_COLUMNS = []
for cols in SCORE_GROUPS.values():
    ALL_SCORE_COLUMNS.extend(cols)

# ─────────────────────────── LightGBM: Frequency (классификация) ──────
FREQ_PARAMS_LGB = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 1000,
    "verbose": -1,
    "random_state": RANDOM_SEED,
    "is_unbalance": True,
}

# ─────────────────────────── LightGBM: Severity (регрессия) ───────────
SEV_PARAMS_LGB = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_estimators": 800,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────── Sklearn GradientBoosting (fallback) ──────
FREQ_PARAMS_SKL = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_leaf": 50,
    "subsample": 0.8,
    "max_features": 0.8,
    "random_state": RANDOM_SEED,
}

SEV_PARAMS_SKL = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_leaf": 30,
    "subsample": 0.8,
    "max_features": 0.8,
    "loss": "squared_error",
    "random_state": RANDOM_SEED,
}

# ─────────────────────────── Выбор параметров по бэкенду ──────────────
if MODEL_BACKEND == "lightgbm":
    FREQ_PARAMS = FREQ_PARAMS_LGB
    SEV_PARAMS = SEV_PARAMS_LGB
else:
    FREQ_PARAMS = FREQ_PARAMS_SKL
    SEV_PARAMS = SEV_PARAMS_SKL

# ─────────────────────────── WoE ──────────────────────────────────────
WOE_FEATURES = ["experience_bucket", "engine_power_bucket", "bonus_malus_bucket"]
WOE_REGULARIZATION = 0.5

# ─────────────────────────── Оптимизация ──────────────────────────────
OPTIM_MARGIN_GRID_POINTS = 200
OPTIM_MARGIN_RANGE = (0.5, 2.5)

print(f"[CONFIG] Backend: {MODEL_BACKEND}")
