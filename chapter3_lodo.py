"""
ВКР — Глава 3. БЛОК 2 ТЗ: Leave-One-Dataset-Out (LODO) валидация advisor-а.

Назначение
──────────
До этого момента advisor оценивался на тех же 27 датасетах, на которых
обучен KnowledgeBase. Это — circular validation, и на защите по нему будет
методологическое замечание: «вы валидируете на тех же данных, на которых
обучаете». Настоящий модуль закрывает этот пробел.

LODO-протокол:
  Для каждого датасета D_i ∈ {D_1, ..., D_27}:
    1. Строим KB_{-i} — KnowledgeBase из 26 датасетов БЕЗ D_i;
    2. Для каждой пары (модель, преобразование) из advisor-рекомендации:
       — берём predicted_delta_pct из KB_{-i} (медиана по бину);
       — берём actual_delta из строки D_i в полном CSV (это «истина»);
       — записываем (predicted, actual, abs_error, sign_match, rank).
    3. Вычисляем агрегаты по всему набору пар:
       — MAE_delta       — средняя абсолютная ошибка прогноза ΔRMSE;
       — top1_hit_rate   — доля случаев, когда top-1 рекомендация
                          действительно оказалась лучшим преобразованием;
       — sign_match_rate — доля случаев совпадения знака прогноза и факта;
       — spearman_corr   — корреляция Спирмена предсказанных и фактических Δ.

Выходные артефакты
──────────────────
  advisor_output/lodo_validation.csv    — все пары с предсказаниями и фактами
  advisor_output/lodo_summary.json      — агрегированные метрики
  advisor_output/lodo_by_bin.csv        — разбивка MAE по γ₁-бинам

Использование
─────────────
    python chapter3_lodo.py                                     # ищет full_results_v6.csv
    python chapter3_lodo.py --csv results/full_results_v6.csv

API:
    from chapter3_lodo import lodo_validate
    df_lodo, summary = lodo_validate("results/full_results_v6.csv")
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from chapter3_advisor_v3 import (
        KnowledgeBase, gamma_bin_of, LABEL_TO_TR, LABEL_TO_MODEL,
        GAMMA_BINS, ADVISOR_DIR,
    )
    from chapter2_experiments_v6 import (
        MODELS, MODEL_LABEL, REGRESSION_MODELS, TRANSFORMS, TR_LABEL, ALPHA,
    )
except ImportError as e:
    raise ImportError(
        "chapter3_lodo.py требует chapter2_experiments_v6.py и "
        f"chapter3_advisor_v3.py в той же директории: {e}")

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# 1. ХОЛДОУТ KB
# ═════════════════════════════════════════════════════════════════════════════

def build_kb_excluding(full_csv_path: Path, hold_out_dataset: str
                       ) -> KnowledgeBase:
    """Строит KnowledgeBase из CSV, исключая один датасет.

    Параметры
    ─────────
    full_csv_path    : путь к full_results_v6.csv (или v5)
    hold_out_dataset : имя датасета для исключения (значение колонки 'Dataset')

    Возвращает KnowledgeBase, полученный без строк hold_out_dataset.
    Технически: читаем CSV, фильтруем, временно сохраняем во временный CSV,
    вызываем from_csv. (Прямой in-memory путь усложнил бы interface KB.)
    """
    df = pd.read_csv(full_csv_path)
    if "Dataset" not in df.columns:
        raise ValueError(f"{full_csv_path}: нет колонки 'Dataset'")
    n_before = df["Dataset"].nunique()
    df_filtered = df[df["Dataset"] != hold_out_dataset].copy()
    n_after = df_filtered["Dataset"].nunique()
    if n_after >= n_before:
        warnings.warn(
            f"build_kb_excluding('{hold_out_dataset}'): датасет не найден "
            f"в CSV; результат идентичен полному KB.")

    # Временный CSV → KnowledgeBase.from_csv. Это самый простой вариант,
    # учитывая что from_csv делает много логики (бины, sig_better_rate и т.д.).
    tmp_path = ADVISOR_DIR / f"_tmp_kb_excl_{hash(hold_out_dataset) & 0xFFFF}.csv"
    df_filtered.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    try:
        kb = KnowledgeBase.from_csv(tmp_path)
        kb.source = f"LODO: excl='{hold_out_dataset}'"
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return kb


# ═════════════════════════════════════════════════════════════════════════════
# 2. ОСНОВНАЯ LODO-ВАЛИДАЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

def lodo_validate(full_csv_path: str,
                  model_keys: Tuple[str, ...] = ("linear", "ridge", "lasso"),
                  transforms_to_test: Optional[List[str]] = None,
                  verbose: bool = True
                  ) -> Tuple[pd.DataFrame, dict]:
    """LODO-валидация advisor-а.

    Параметры
    ─────────
    full_csv_path     : путь к full_results_v6.csv (или v5).
    model_keys        : какие модели проверять (по умолчанию — линейные;
                        для GLM/spec-loss потребуется наличие соответствующих
                        строк в CSV).
    transforms_to_test: список преобразований; None → все, кроме 'none'.
    verbose           : печать прогресса.

    Возвращает
    ──────────
    (df_lodo, summary):
        df_lodo : DataFrame со строками-парами (dataset, model, transform)
                  с колонками predicted_delta, actual_delta, abs_error,
                  sign_match, rank_predicted, rank_actual, gamma_bin.
        summary : dict с агрегированными метриками (MAE_delta,
                  top1_hit_rate, sign_match_rate, spearman_correlation,
                  + те же метрики в разбивке по бинам).
    """
    full_csv_path = Path(full_csv_path)
    if not full_csv_path.exists():
        raise FileNotFoundError(f"Не найден CSV с результатами: {full_csv_path}")

    df_full = pd.read_csv(full_csv_path)
    needed = ["Dataset", "gamma1", "Model", "Transform", "delta_RMSE_test_pct"]
    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют колонки: {missing}")

    # Маппинг человекочитаемых имён → внутренние ключи
    df_full["Model_key"]     = df_full["Model"].map(LABEL_TO_MODEL)
    df_full["Transform_key"] = df_full["Transform"].map(LABEL_TO_TR)
    df_full["gamma_bin"]     = df_full["gamma1"].apply(gamma_bin_of)

    datasets = sorted(df_full["Dataset"].unique())
    if transforms_to_test is None:
        transforms_to_test = [t for t in TRANSFORMS if t != "none"]
    if verbose:
        print(f"\n  LODO-валидация: {len(datasets)} датасетов, "
              f"модели={list(model_keys)}, "
              f"преобр.={transforms_to_test}")

    records = []
    for i, hold_out in enumerate(datasets, start=1):
        if verbose:
            print(f"    [{i}/{len(datasets)}] hold-out = {hold_out}")

        kb_subset = build_kb_excluding(full_csv_path, hold_out)
        df_holdout = df_full[df_full["Dataset"] == hold_out].copy()
        if df_holdout.empty:
            continue
        gamma1_h = float(df_holdout["gamma1"].iloc[0])
        gbin_h = gamma_bin_of(gamma1_h)

        # ── Для каждой модели берём предсказания из KB_{-i} и сравниваем ──
        for m_key in model_keys:
            # Прогноз: получаем рекомендованные преобразования из KB_{-i}
            kb_for_model = kb_subset._rules[
                (kb_subset._rules["Model_key"] == m_key) &
                (kb_subset._rules["gamma_bin"] == gbin_h)
            ].copy().sort_values("median_delta_pct").reset_index(drop=True)

            if kb_for_model.empty:
                # Нет данных в KB_{-i} для этого (model, bin) → пропускаем
                continue

            # Фактический ranking по datu_holdout (среди тестируемых преобр.)
            df_holdout_m = df_holdout[
                (df_holdout["Model_key"] == m_key) &
                (df_holdout["Transform_key"].isin(transforms_to_test))
            ].copy()
            if df_holdout_m.empty:
                continue
            df_holdout_m = df_holdout_m.sort_values(
                "delta_RMSE_test_pct").reset_index(drop=True)
            df_holdout_m["rank_actual"] = (
                df_holdout_m["delta_RMSE_test_pct"].rank(method="min").astype(int))

            # Для каждого преобразования: predicted vs actual
            for _, kb_row in kb_for_model.iterrows():
                tr = kb_row["Transform_key"]
                if tr not in transforms_to_test:
                    continue

                pred = float(kb_row["median_delta_pct"])
                rank_pred = int(kb_row["rank"])

                row_h = df_holdout_m[df_holdout_m["Transform_key"] == tr]
                if row_h.empty:
                    continue
                actual = float(row_h["delta_RMSE_test_pct"].iloc[0])
                rank_actual = int(row_h["rank_actual"].iloc[0])

                records.append({
                    "dataset":     hold_out,
                    "gamma1":      gamma1_h,
                    "gamma_bin":   gbin_h,
                    "model":       MODEL_LABEL.get(m_key, m_key),
                    "model_key":   m_key,
                    "transform":   TR_LABEL.get(tr, tr),
                    "transform_key": tr,
                    "predicted_delta_pct": pred,
                    "actual_delta_pct":    actual,
                    "abs_error":   abs(actual - pred),
                    "sign_match":  int(
                        np.sign(actual) == np.sign(pred)
                        and abs(actual) > 0.5  # игнорируем «почти ноль»
                    ),
                    "rank_predicted": rank_pred,
                    "rank_actual":    rank_actual,
                    "top1_predicted": int(rank_pred == 1),
                    "top1_actual":    int(rank_actual == 1),
                    "is_top1_hit":    int(rank_pred == 1 and rank_actual == 1),
                })

    if not records:
        raise RuntimeError(
            "LODO не построил ни одной пары — проверьте, что CSV содержит "
            "все нужные пары (model × transform) для модели(ей) из "
            f"{model_keys}.")

    df_lodo = pd.DataFrame(records)

    # ── АГРЕГАТЫ ─────────────────────────────────────────────────────────────
    summary = _compute_lodo_metrics(df_lodo, verbose=verbose)

    return df_lodo, summary


def _compute_lodo_metrics(df_lodo: pd.DataFrame, verbose: bool = True) -> dict:
    """Считает агрегаты MAE_delta, top1_hit_rate, sign_match_rate,
    spearman_correlation — по всему массиву и в разбивке по γ₁-бинам.
    """
    from scipy.stats import spearmanr

    def _block_metrics(df_block: pd.DataFrame) -> dict:
        if df_block.empty:
            return {"n_pairs": 0,
                    "MAE_delta": float("nan"),
                    "top1_hit_rate": float("nan"),
                    "sign_match_rate": float("nan"),
                    "spearman_correlation": float("nan"),
                    "spearman_p": float("nan")}
        mae = float(df_block["abs_error"].mean())
        # top1_hit_rate: среди пар, где rank_predicted == 1
        df_top1 = df_block[df_block["top1_predicted"] == 1]
        if not df_top1.empty:
            top1_hit = float(df_top1["is_top1_hit"].mean())
        else:
            top1_hit = float("nan")
        sign = float(df_block["sign_match"].mean())
        try:
            rho, p_rho = spearmanr(df_block["predicted_delta_pct"],
                                   df_block["actual_delta_pct"])
            rho, p_rho = float(rho), float(p_rho)
        except Exception:
            rho, p_rho = float("nan"), float("nan")
        return {
            "n_pairs": int(len(df_block)),
            "MAE_delta": mae,
            "top1_hit_rate": top1_hit,
            "sign_match_rate": sign,
            "spearman_correlation": rho,
            "spearman_p": p_rho,
        }

    overall = _block_metrics(df_lodo)

    # По бинам
    by_bin = {}
    for gbin in GAMMA_BINS:
        df_b = df_lodo[df_lodo["gamma_bin"] == gbin]
        by_bin[gbin] = _block_metrics(df_b)

    # По моделям
    by_model = {}
    for m_label in df_lodo["model"].unique():
        df_m = df_lodo[df_lodo["model"] == m_label]
        by_model[m_label] = _block_metrics(df_m)

    summary = {
        "overall": overall,
        "by_gamma_bin": by_bin,
        "by_model": by_model,
        "n_datasets": int(df_lodo["dataset"].nunique()),
        "n_models":   int(df_lodo["model"].nunique()),
        "n_transforms": int(df_lodo["transform"].nunique()),
    }

    if verbose:
        print("\n  ─" * 35)
        print(f"  LODO-метрики (всего {overall['n_pairs']} пар)")
        print(f"    MAE прогноза ΔRMSE       = {overall['MAE_delta']:.2f} pp")
        print(f"    top-1 hit rate           = {overall['top1_hit_rate']*100:.1f}%")
        print(f"    sign-match rate          = {overall['sign_match_rate']*100:.1f}%")
        print(f"    Spearman ρ (pred,actual) = {overall['spearman_correlation']:+.3f}"
              f"  (p = {overall['spearman_p']:.4g})")

        print("\n  По бинам γ₁:")
        print(f"    {'Бин':<40} {'n':>5} {'MAE':>7} {'top1':>7} {'sign':>7} {'ρ':>7}")
        for gbin, m in by_bin.items():
            print(f"    {gbin:<40} {m['n_pairs']:>5} "
                  f"{m['MAE_delta']:>7.2f} "
                  f"{m['top1_hit_rate']*100:>6.1f}% "
                  f"{m['sign_match_rate']*100:>6.1f}% "
                  f"{m['spearman_correlation']:>+7.3f}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 3. СОХРАНЕНИЕ АРТЕФАКТОВ
# ═════════════════════════════════════════════════════════════════════════════

def save_lodo_artifacts(df_lodo: pd.DataFrame, summary: dict,
                        out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Сохраняет lodo_validation.csv, lodo_summary.json, lodo_by_bin.csv."""
    out_dir = Path(out_dir) if out_dir else ADVISOR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    csv_path = out_dir / "lodo_validation.csv"
    df_lodo.to_csv(csv_path, index=False, encoding="utf-8-sig")
    paths["csv"] = csv_path

    json_path = out_dir / "lodo_summary.json"
    # NaN → None в JSON
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(summary), f, ensure_ascii=False, indent=2)
    paths["json"] = json_path

    by_bin_rows = [{"gamma_bin": gbin, **m}
                   for gbin, m in summary["by_gamma_bin"].items()]
    by_bin_path = out_dir / "lodo_by_bin.csv"
    pd.DataFrame(by_bin_rows).to_csv(by_bin_path, index=False,
                                      encoding="utf-8-sig")
    paths["by_bin_csv"] = by_bin_path

    return paths


# ═════════════════════════════════════════════════════════════════════════════
# 4. CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LODO-валидация advisor-а из Главы 3.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Путь к full_results_v6.csv "
                             "(по умолчанию ищется в results/).")
    parser.add_argument("--models", type=str, default="linear,ridge,lasso",
                        help="Модели для LODO через запятую "
                             "(по умолчанию: linear,ridge,lasso).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Куда сохранить артефакты "
                             f"(по умолчанию: {ADVISOR_DIR}).")
    args = parser.parse_args()

    csv_candidates = []
    if args.csv:
        csv_candidates.append(Path(args.csv))
    csv_candidates += [
        Path("results/full_results_v6.csv"),
        Path("results/full_results_v5.csv"),
        Path("results/full_results_v4.csv"),
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            "Не найден CSV с результатами Главы 2. "
            "Сначала запустите chapter2_experiments_v6.py.")

    print("\n" + "█" * 72)
    print("  ВКР — ГЛАВА 3, БЛОК 2: LODO-ВАЛИДАЦИЯ ADVISOR-А")
    print(f"  Источник: {csv_path}")
    print("█" * 72)

    model_keys = tuple(m.strip() for m in args.models.split(",") if m.strip())
    df_lodo, summary = lodo_validate(str(csv_path), model_keys=model_keys,
                                      verbose=True)

    out_dir = Path(args.out_dir) if args.out_dir else ADVISOR_DIR
    paths = save_lodo_artifacts(df_lodo, summary, out_dir=out_dir)
    print("\n  Артефакты:")
    for k, p in paths.items():
        print(f"    {k}: {p}")
    print("\n  ✓ LODO-валидация завершена.\n")


if __name__ == "__main__":
    main()
