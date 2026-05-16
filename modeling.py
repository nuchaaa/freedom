"""
Модуль моделирования (Frequency-Severity подход).
Адаптирован под реальный датасет ОГПО с динамическим списком признаков.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Tuple, Dict, Any, Optional

from config import (
    FREQ_PARAMS, SEV_PARAMS, RANDOM_SEED, TEST_SIZE, MODEL_BACKEND,
    COL_IS_CLAIM, COL_CLAIM_AMOUNT,
)


def _create_classifier(params):
    if MODEL_BACKEND == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**params)


def _create_regressor(params):
    if MODEL_BACKEND == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params)


def split_data(df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """Стратифицированное разбиение по is_claim."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[COL_IS_CLAIM]
    )
    print(f"[SPLIT] Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"[SPLIT] Claim Rate — Train: {train_df[COL_IS_CLAIM].mean():.2%} | Test: {test_df[COL_IS_CLAIM].mean():.2%}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class FrequencyModel:
    """Бинарная классификация P(ДТП) с auto scale_pos_weight."""

    def __init__(self, params=None):
        self.params = params or FREQ_PARAMS.copy()
        self.model = None
        self.feature_importance_df = None

    def fit(self, train_df, val_df=None, features=None, target=COL_IS_CLAIM):
        X_train = train_df[features]
        y_train = train_df[target]

        if MODEL_BACKEND == "sklearn":
            n_neg = (y_train == 0).sum()
            n_pos = max((y_train == 1).sum(), 1)
            sw = np.where(y_train == 1, n_neg / n_pos, 1.0)
            print(f"[FREQ] scale_pos_weight: {n_neg / n_pos:.2f}")
        else:
            sw = None

        self.model = _create_classifier(self.params)

        if MODEL_BACKEND == "lightgbm" and val_df is not None:
            try:
                import lightgbm as lgb
                self.model.fit(X_train, y_train,
                               eval_set=[(val_df[features], val_df[target])],
                               callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)])
            except ImportError:
                self.model.fit(X_train, y_train, sample_weight=sw)
        else:
            self.model.fit(X_train, y_train, sample_weight=sw)

        self.feature_importance_df = pd.DataFrame({
            "feature": features, "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"[FREQ] Модель обучена ({MODEL_BACKEND})")
        return self

    def predict_proba(self, df, features):
        return self.model.predict_proba(df[features])[:, 1]

    def evaluate(self, df, features, target=COL_IS_CLAIM):
        y_true = df[target].values
        y_pred = self.predict_proba(df, features)
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        print(f"[FREQ] ROC-AUC: {auc:.4f} | GINI: {gini:.4f}")
        return {"roc_auc": auc, "gini": gini}

    def get_top_features(self, n=10):
        return self.feature_importance_df.head(n)


class SeverityModel:
    """Регрессия размера выплаты. Обучается только на is_claim==1."""

    def __init__(self, params=None):
        self.params = params or SEV_PARAMS.copy()
        self.model = None
        self.feature_importance_df = None

    def fit(self, train_df, val_df=None, features=None, target=COL_CLAIM_AMOUNT):
        train_claims = train_df[train_df[COL_IS_CLAIM] == 1]
        if len(train_claims) == 0:
            raise ValueError("Нет строк с выплатами!")

        X_train = train_claims[features]
        y_train = train_claims[target]
        self.model = _create_regressor(self.params)

        if MODEL_BACKEND == "lightgbm" and val_df is not None:
            try:
                import lightgbm as lgb
                val_claims = val_df[val_df[COL_IS_CLAIM] == 1]
                if len(val_claims) > 0:
                    self.model.fit(X_train, y_train,
                                   eval_set=[(val_claims[features], val_claims[target])],
                                   callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)])
                else:
                    self.model.fit(X_train, y_train)
            except ImportError:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        self.feature_importance_df = pd.DataFrame({
            "feature": features, "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"[SEV] Обучена на {len(train_claims):,} полисах. Средн. выплата: {y_train.mean():,.0f} ₸")
        return self

    def predict(self, df, features):
        return np.clip(self.model.predict(df[features]), 0, None)

    def evaluate(self, df, features, target=COL_CLAIM_AMOUNT):
        claims = df[df[COL_IS_CLAIM] == 1]
        if len(claims) == 0:
            return {"rmse": np.nan, "mae": np.nan}
        y_true = claims[target].values
        y_pred = self.predict(claims, features)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        print(f"[SEV] RMSE: {rmse:,.0f} ₸ | MAE: {mae:,.0f} ₸")
        return {"rmse": rmse, "mae": mae}

    def get_top_features(self, n=10):
        return self.feature_importance_df.head(n)


def compute_pure_premium(df, freq_model, sev_model, features):
    """Pure Premium = P(ДТП) × E[Severity | ДТП]"""
    df = df.copy()
    df["prob_claim"] = freq_model.predict_proba(df, features)
    df["predicted_severity"] = sev_model.predict(df, features)
    df["pure_premium"] = df["prob_claim"] * df["predicted_severity"]
    print(f"[PP] Avg P(ДТП): {df['prob_claim'].mean():.4f} | "
          f"Avg Severity: {df['predicted_severity'].mean():,.0f} | "
          f"Avg Pure Premium: {df['pure_premium'].mean():,.0f} ₸")
    return df


def train_pipeline(df, feature_cols):
    """Полный pipeline: split → frequency → severity → pure premium."""
    print(f"\n{'=' * 70}\n  TRAINING PIPELINE ({MODEL_BACKEND.upper()})\n{'=' * 70}")

    train_df, test_df = split_data(df)

    print("\n── Frequency Model ────────────────────────────")
    freq = FrequencyModel()
    freq.fit(train_df, val_df=test_df, features=feature_cols)
    print("  [Train]", end=" "); freq_train = freq.evaluate(train_df, feature_cols)
    print("  [Test] ", end=" "); freq_test = freq.evaluate(test_df, feature_cols)

    print("\n── Severity Model ─────────────────────────────")
    sev = SeverityModel()
    sev.fit(train_df, val_df=test_df, features=feature_cols)
    sev_metrics = sev.evaluate(test_df, feature_cols)

    print("\n── Pure Premium ───────────────────────────────")
    test_df = compute_pure_premium(test_df, freq, sev, feature_cols)

    print("\n── Top-10 Features (Frequency) ─────────────────")
    print(freq.get_top_features(10).to_string(index=False))

    metrics = {"frequency": {"train": freq_train, "test": freq_test}, "severity": sev_metrics}
    return test_df, freq, sev, metrics
