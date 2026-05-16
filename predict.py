"""
Скрипт для получения предсказаний на тестовой выборке (test.csv).
Запускает обучение на train, сохраняет модель и делает предсказания для test.
"""

import sys
import pickle
import pandas as pd
from data_preparation import prepare_dataset, load_data, generate_synthetic_real_format
from pricing_optimization import compute_new_premium
from main import main as run_train_pipeline
from config import COL_CONTRACT, COL_PREMIUM, COL_IS_CLAIM, COL_CLAIM_AMOUNT

def main():
    if len(sys.argv) < 3:
        print("Использование: python3 predict.py <путь_к_train.csv> <путь_к_test.csv>")
        print("Использую демо-режим (синтетические данные).")
        train_path = None
        test_path = None
    else:
        train_path = sys.argv[1]
        test_path = sys.argv[2]

    # 1. Запускаем полный пайплайн на Train
    print("\n" + "="*50)
    print("1. ОБУЧЕНИЕ МОДЕЛИ НА TRAIN DATASET")
    print("="*50)
    df_train, freq_model, sev_model, woe_transformer, best_margin, feature_cols = run_train_pipeline(train_path)

    # 2. Сохраняем модель в файл
    print("\n" + "="*50)
    print("2. ЭКСПОРТ МОДЕЛИ (ML-модель)")
    print("="*50)
    model_artifact = {
        "frequency_model": freq_model,
        "severity_model": sev_model,
        "woe_transformer": woe_transformer,
        "features": feature_cols,
        "best_margin": best_margin
    }
    with open("ogpo_model.pkl", "wb") as f:
        pickle.dump(model_artifact, f)
    print("[EXPORT] Модели и параметры сохранены в файл: ogpo_model.pkl")

    # 3. Готовим Test данные (используя обученный woe_transformer)
    print("\n" + "="*50)
    print("3. ОБРАБОТКА TEST DATASET")
    print("="*50)
    
    if test_path:
        try:
            raw_test = load_data(test_path)
        except FileNotFoundError:
            print(f"[WARN] Файл '{test_path}' не найден! Генерируем тестовую синтетику...")
            raw_test = generate_synthetic_real_format(n_policies=10000, n_drivers=15000, seed=42)
            # Убираем таргеты, чтобы имитировать реальный test
            if COL_IS_CLAIM in raw_test.columns: raw_test.drop(columns=[COL_IS_CLAIM], inplace=True)
            if COL_CLAIM_AMOUNT in raw_test.columns: raw_test.drop(columns=[COL_CLAIM_AMOUNT], inplace=True)
    else:
        raw_test = generate_synthetic_real_format(n_policies=10000, n_drivers=15000, seed=42)
        if COL_IS_CLAIM in raw_test.columns: raw_test.drop(columns=[COL_IS_CLAIM], inplace=True)
        if COL_CLAIM_AMOUNT in raw_test.columns: raw_test.drop(columns=[COL_CLAIM_AMOUNT], inplace=True)

    # Применяем подготовку данных (агрегация + фичи) с тем же WoE
    df_test, _, _ = prepare_dataset(path=None if not test_path else test_path, woe_transformer=woe_transformer)

    # Проверяем, что все признаки есть (если чего-то нет - заполняем 0 или средним)
    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = 0.0

    # 4. Делаем предсказания
    print("\n" + "="*50)
    print("4. ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ")
    print("="*50)
    df_test["prob_claim"] = freq_model.predict_proba(df_test, feature_cols)
    df_test["predicted_severity"] = sev_model.predict(df_test, feature_cols)
    df_test["pure_premium"] = df_test["prob_claim"] * df_test["predicted_severity"]

    # Рассчитываем новую цену (с оптимизированным маржинальным коэффициентом)
    df_test = compute_new_premium(df_test, margin_coeff=best_margin)

    # 5. Сохраняем submission файл
    submission_cols = [
        COL_CONTRACT,
        "prob_claim",          # вероятность попадания в ДТП
        "predicted_severity",  # прогноз убытка (если будет ДТП)
        "pure_premium",        # мат. ожидание убытка (прогноз убыточности)
        "new_premium"          # новая стоимость полиса
    ]
    
    # Добавляем старую премию для наглядности (если она есть)
    if COL_PREMIUM in df_test.columns:
        submission_cols.insert(1, COL_PREMIUM)

    # Убеждаемся, что все колонки есть
    final_cols = [c for c in submission_cols if c in df_test.columns]
    
    submission_df = df_test[final_cols]
    submission_df.to_csv("submission.csv", index=False)
    
    print("\n[SUCCESS] Предсказания завершены!")
    print(f"Сгенерирован файл: submission.csv ({len(submission_df)} строк)")
    print("\nПример первых 5 строк submission.csv:")
    print(submission_df.head().to_string(index=False))

if __name__ == "__main__":
    main()
