"""
ВКР — Глава 4. БЛОК 6 ТЗ: Benchmark advisor vs FLAML.

Назначение
──────────
Закрывает методологический вопрос «зачем нужен advisor, если есть FLAML /
AutoGluon, который автоматически найдёт лучшую модель за минуту».

Постановка эксперимента
───────────────────────
Для каждого датасета из выбранного списка прогоняем ДВА пути:

  Путь 1 (Advisor):
    a) diagnose_target(y) — без обучения модели;
    b) recommend(y, "linear", kb) — берём top-1 рекомендованное преобразование;
    c) обучаем линейную модель с этим преобразованием на train;
    d) измеряем RMSE на hold-out test, фиксируем суммарное время.

  Путь 2 (FLAML):
    a) fit AutoML(time_budget=60s) на train без advisor-а;
    b) измеряем RMSE на том же hold-out test, фиксируем время.

Метрики
───────
  • rmse_advisor, rmse_flaml, rmse_ratio = advisor / flaml;
  • t_advice_ms, t_advisor_total_s, t_flaml_s, speedup = t_flaml / t_advice;
  • advisor_recommendation (какое преобразование), flaml_best_model.

Результат интерпретируется как:
  «advisor даёт X% от качества FLAML за Y% времени, плюс объяснимость и
  возможность ручной верификации».

Использование
─────────────
    python chapter4_benchmark.py                # стандартные 5 датасетов
    python chapter4_benchmark.py --time-budget 30
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from chapter3_advisor_v3 import (
        KnowledgeBase, recommend, diagnose_target, ADVISOR_DIR,
    )
    from chapter2_experiments_v6 import (
        apply_transform, make_model, general_smearing,
        RANDOM_STATE, TEST_SIZE, RESULTS_DIR, TR_LABEL,
    )
except ImportError as e:
    raise ImportError(
        f"chapter4_benchmark.py требует chapter2_experiments_v6.py + "
        f"chapter3_advisor_v3.py в той же директории: {e}")

from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

BENCHMARK_DIR = Path("chapter4_benchmark")
BENCHMARK_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Список loader-ов для бенчмарка (5 датасетов с разным γ₁ по ТЗ)
# ─────────────────────────────────────────────────────────────────────────────
#
# По ТЗ:
#   Concrete (γ₁ ≈ 0.4) — симметричный кейс
#   Auto MPG (γ₁ ≈ 0.5) — на границе
#   Boston Housing (γ₁ ≈ 1.1) — умеренная асимметрия
#   Medical Insurance (γ₁ ≈ 1.5) — сильная
#   Yacht Hydrodynamics (γ₁ ≈ 3.0) — экстремальная

def _load_concrete():
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=4353, as_frame=True, parser="auto")
        df = d.frame.dropna()
        target = d.target.name if d.target is not None else df.columns[-1]
        X = df.drop(columns=[target]).select_dtypes(include="number").values.astype(float)
        y = df[target].values.astype(float)
        return X, y, "Concrete"
    except Exception as e:
        return None, None, f"Concrete (skip: {type(e).__name__})"


def _load_auto_mpg():
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name="autoMpg", version=1, as_frame=True, parser="auto")
        df = d.frame.dropna()
        # target обычно 'class' или 'mpg'
        target = None
        for cand in ("mpg", "class"):
            if cand in df.columns:
                target = cand; break
        if target is None:
            target = df.columns[-1]
        # Удаляем нечисловые
        Xdf = df.drop(columns=[target]).copy()
        for c in Xdf.columns:
            if not pd.api.types.is_numeric_dtype(Xdf[c]):
                Xdf[c] = pd.factorize(Xdf[c])[0]
        Xdf = Xdf.apply(pd.to_numeric, errors="coerce").dropna()
        ydf = df.loc[Xdf.index, target]
        return Xdf.values.astype(float), ydf.astype(float).values, "AutoMPG"
    except Exception as e:
        return None, None, f"AutoMPG (skip: {type(e).__name__})"


def _load_boston():
    """Загружает Boston Housing (через OpenML — sklearn убрал её)."""
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
        df = d.frame.dropna()
        target = "MEDV" if "MEDV" in df.columns else df.columns[-1]
        Xdf = df.drop(columns=[target])
        for c in Xdf.columns:
            if not pd.api.types.is_numeric_dtype(Xdf[c]):
                Xdf[c] = pd.factorize(Xdf[c])[0]
        return Xdf.values.astype(float), df[target].values.astype(float), "Boston"
    except Exception as e:
        return None, None, f"Boston (skip: {type(e).__name__})"


def _load_medical_insurance():
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name="medical_charges", version=1, as_frame=True,
                         parser="auto")
        df = d.frame.dropna()
        target = "AverageTotalPayments" if "AverageTotalPayments" in df.columns \
                 else df.columns[-1]
        Xdf = df.drop(columns=[target])
        for c in Xdf.columns:
            if not pd.api.types.is_numeric_dtype(Xdf[c]):
                Xdf[c] = pd.factorize(Xdf[c])[0]
        return Xdf.values.astype(float), df[target].values.astype(float), \
               "MedicalInsurance"
    except Exception as e:
        return None, None, f"MedicalInsurance (skip: {type(e).__name__})"


def _load_yacht():
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name="yacht_hydrodynamics", version=1,
                         as_frame=True, parser="auto")
        df = d.frame.dropna()
        target = df.columns[-1]
        Xdf = df.drop(columns=[target])
        for c in Xdf.columns:
            if not pd.api.types.is_numeric_dtype(Xdf[c]):
                Xdf[c] = pd.factorize(Xdf[c])[0]
        return Xdf.values.astype(float), df[target].values.astype(float), "Yacht"
    except Exception as e:
        return None, None, f"Yacht (skip: {type(e).__name__})"


def _load_synthetic_fallback(name: str, target_skew: float, n: int = 2000):
    """Резерв: лог-нормальная синтетика с заданной асимметрией."""
    rng = np.random.default_rng(hash(name) & 0xFFFF)
    X = rng.standard_normal((n, 5))
    # σ подобрана так, чтобы при экспоненцировании дать ~target_skew
    sigma = max(0.1, target_skew * 0.5)
    log_y = 1.0 + X @ np.array([0.3, 0.2, -0.1, 0.05, 0.01]) + \
            sigma * rng.standard_normal(n)
    return X, np.exp(log_y), f"{name}_synth"


BENCHMARK_DATASETS: list = [
    # (name, loader, target_skew_fallback)
    ("Concrete",         _load_concrete,         0.4),
    ("AutoMPG",          _load_auto_mpg,         0.5),
    ("Boston",           _load_boston,           1.1),
    ("MedicalInsurance", _load_medical_insurance, 1.5),
    ("Yacht",            _load_yacht,            3.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Тренировка с преобразованием Y и smearing-инверсией
# ─────────────────────────────────────────────────────────────────────────────

def _train_linear_with_transform(X_tr, y_tr, X_te, y_te, transform: str,
                                 shift: float = 0.0
                                 ) -> Tuple[float, np.ndarray]:
    """Обучает LinearRegression с заданным преобразованием Y; возвращает
    (RMSE на test в исходной шкале, прогноз на test).

    Использует общую smearing-коррекцию (Duan 1983) для нелинейных монотонных g.
    """
    # Подготовка преобразования
    SMEAR_APPLICABLE = {"log", "sqrt", "asinh", "boxcox", "yeojohnson"}
    try:
        y_tr_t, y_te_t, lam, inv = apply_transform(y_tr, y_te, transform,
                                                    shift=shift)
    except Exception as e:
        # Преобразование не применимо (например, log на отриц.) —
        # падаем на baseline
        warnings.warn(f"Преобразование '{transform}' не применилось: "
                      f"{type(e).__name__}; используется 'none'.")
        return _train_linear_with_transform(X_tr, y_tr, X_te, y_te,
                                             "none", shift=shift)

    model = LinearRegression()
    model.fit(X_tr, y_tr_t)
    pred_t_te = model.predict(X_te)

    if transform in SMEAR_APPLICABLE:
        pred_t_tr = model.predict(X_tr)
        resid_tr = y_tr_t - pred_t_tr
        pred_orig = general_smearing(pred_t_te, resid_tr, inv)
    else:
        pred_orig = inv(pred_t_te)

    pred_orig = np.nan_to_num(pred_orig, nan=0.0, posinf=1e12, neginf=0.0)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred_orig)))
    return rmse, pred_orig


# ─────────────────────────────────────────────────────────────────────────────
# FLAML AutoML
# ─────────────────────────────────────────────────────────────────────────────

def _run_flaml(X_tr, y_tr, X_te, y_te, time_budget: int = 60
               ) -> Tuple[float, str, np.ndarray]:
    """Запускает FLAML AutoML, возвращает (RMSE на test, имя выбранного
    estimator-а, прогноз на test).

    Если FLAML не установлен — пробуем fallback на RandomForest / XGBoost
    с дефолтными гиперпараметрами как «прокси для AutoML», но честнее
    пометить это в результате.
    """
    try:
        from flaml import AutoML
    except ImportError:
        warnings.warn("FLAML не установлен. Использую XGBoost как прокси для "
                      "AutoML (НЕ-настоящий бенчмарк); pip install flaml[automl]")
        import xgboost as xgb
        m = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        return rmse, "xgb_proxy (FLAML not installed)", pred

    automl = AutoML()
    automl_settings = {
        "time_budget": time_budget,
        "metric": "rmse",
        "task": "regression",
        "log_file_name": "",         # отключаем лог
        "verbose": 0,
        # Регрессионные оценщики FLAML: lgbm, xgboost, rf, extra_tree.
        # 'lrl1' (LogisticRegression) — только для классификации.
        "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree"],
    }
    automl.fit(X_train=X_tr, y_train=y_tr, **automl_settings)
    pred = automl.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    best = getattr(automl, "best_estimator", "unknown")
    return rmse, str(best), pred


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one_dataset(name: str,
                          loader: Callable,
                          fallback_skew: float,
                          kb: KnowledgeBase,
                          time_budget: int = 60,
                          use_synthetic_fallback: bool = True
                          ) -> dict:
    """Прогоняет один датасет через advisor и FLAML, возвращает row-результат."""
    print(f"\n  ── {name} ──")
    X, y, used_name = loader()
    if X is None or y is None:
        if use_synthetic_fallback:
            print(f"    [fallback] загрузка не удалась ({used_name}); "
                  f"генерирую синтетику со skew≈{fallback_skew}.")
            X, y, used_name = _load_synthetic_fallback(name, fallback_skew)
        else:
            return {"dataset": name, "status": "skipped",
                    "reason": used_name}

    # Сабсэмпл для скорости (FLAML на 50k+ строк это долго)
    if len(y) > 5000:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y), 5000, replace=False)
        X, y = X[idx], y[idx]

    g1 = float(skew(y))
    print(f"    n={len(y)}, p={X.shape[1]}, γ₁={g1:+.2f}")

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # ── Path 1: Advisor + рекомендованная модель ─────────────────────────────
    t0 = time.perf_counter()
    diag, recs = recommend(y_tr, "linear", kb=kb, top_k=1, verbose=False)
    top_rec = recs[0] if recs else None
    t_advice = time.perf_counter() - t0
    if top_rec is None:
        return {"dataset": used_name, "status": "advisor_failed"}
    top_transform = top_rec.transform
    print(f"    [advisor] рекомендация: {TR_LABEL.get(top_transform, top_transform)}, "
          f"diag.gamma1={diag.gamma1:+.2f}, t_advice={t_advice*1000:.1f} ms")

    # Обучение с рекомендованным преобразованием
    t0 = time.perf_counter()
    try:
        rmse_advisor, pred_adv = _train_linear_with_transform(
            X_tr, y_tr, X_te, y_te, top_transform,
            shift=diag.suggested_shift)
    except Exception as e:
        print(f"    [advisor] training failed: {e}; используем baseline 'none'")
        rmse_advisor, pred_adv = _train_linear_with_transform(
            X_tr, y_tr, X_te, y_te, "none", shift=0.0)
        top_transform = "none"
    t_train_advisor = time.perf_counter() - t0
    t_advisor_total = t_advice + t_train_advisor
    print(f"    [advisor] RMSE={rmse_advisor:.4g}, "
          f"t_train={t_train_advisor*1000:.1f} ms, "
          f"t_total={t_advisor_total:.3f} s")

    # ── Path 2: FLAML AutoML ────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        rmse_flaml, flaml_best, pred_fl = _run_flaml(
            X_tr, y_tr, X_te, y_te, time_budget=time_budget)
    except Exception as e:
        print(f"    [flaml] FAILED: {e}")
        rmse_flaml, flaml_best, pred_fl = float("nan"), f"error: {type(e).__name__}", None
    t_flaml = time.perf_counter() - t0
    print(f"    [flaml]  RMSE={rmse_flaml:.4g}, best={flaml_best}, "
          f"t={t_flaml:.1f} s")

    # ── Метрики сравнения ───────────────────────────────────────────────────
    if rmse_flaml > 0 and not np.isnan(rmse_flaml):
        rmse_ratio = rmse_advisor / rmse_flaml
    else:
        rmse_ratio = float("nan")
    speedup = (t_flaml / t_advisor_total) if t_advisor_total > 0 else float("nan")

    return {
        "dataset":        used_name,
        "n_test":         int(len(y_te)),
        "n_features":     int(X.shape[1]),
        "gamma1":         g1,
        "advisor_recommendation": TR_LABEL.get(top_transform, top_transform),
        "rmse_advisor":   rmse_advisor,
        "rmse_flaml":     rmse_flaml,
        "rmse_ratio":     rmse_ratio,
        "flaml_best_model": flaml_best,
        "t_advice_ms":    t_advice * 1000.0,
        "t_train_advisor_ms": t_train_advisor * 1000.0,
        "t_advisor_total_s":  t_advisor_total,
        "t_flaml_s":      t_flaml,
        "speedup":        speedup,
        "status":         "ok",
    }


def benchmark_against_flaml(kb: KnowledgeBase,
                            datasets: Optional[list] = None,
                            time_budget: int = 60,
                            use_synthetic_fallback: bool = True
                            ) -> pd.DataFrame:
    """Полный бенчмарк по списку датасетов. Возвращает DataFrame с результатами."""
    if datasets is None:
        datasets = BENCHMARK_DATASETS

    results = []
    for name, loader, fb_skew in datasets:
        try:
            row = benchmark_one_dataset(
                name, loader, fb_skew, kb=kb,
                time_budget=time_budget,
                use_synthetic_fallback=use_synthetic_fallback)
            results.append(row)
        except Exception as e:
            print(f"    [fatal] {name}: {type(e).__name__}: {e}")
            results.append({"dataset": name, "status": f"fatal: {type(e).__name__}"})

    df = pd.DataFrame(results)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Визуализация
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_results(df: pd.DataFrame, out_dir: Path = BENCHMARK_DIR):
    """Строит два графика:
       1) Scatter γ₁ vs rmse_ratio — где advisor близок к FLAML.
       2) Bar chart speedup по датасетам.
    """
    df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df

    if df_ok.empty:
        print("  [plot] нет успешных строк — графики не строятся.")
        return

    # ── (1) γ₁ vs rmse_ratio ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_ok["gamma1"], df_ok["rmse_ratio"], s=120,
               color="#4472C4", edgecolor="black", zorder=3)
    for _, r in df_ok.iterrows():
        ax.annotate(r["dataset"],
                    (r["gamma1"], r["rmse_ratio"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.axhline(1.0, color="#AAAAAA", ls="--", lw=1.2,
               label="advisor = FLAML")
    ax.axhline(1.1, color="#ED7D31", ls=":", lw=1.0,
               label="advisor хуже на 10%")
    ax.set_xlabel(r"$\gamma_1$ (асимметрия Y)")
    ax.set_ylabel(r"RMSE_advisor / RMSE_FLAML")
    ax.set_title("Качество advisor-а относительно FLAML по асимметрии Y")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "benchmark_gamma_vs_rmse_ratio.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → plot: {p1}")

    # ── (2) speedup ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_names = df_ok["dataset"].tolist()
    speedups = df_ok["speedup"].astype(float).tolist()
    bars = ax.bar(ds_names, speedups, color="#70AD47", edgecolor="black")
    for bar, sp in zip(bars, speedups):
        ax.annotate(f"×{sp:,.0f}" if sp >= 1 else f"×{sp:.2f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_ylabel("Speedup (t_flaml / t_advisor_total)")
    ax.set_title("Ускорение: advisor против FLAML")
    ax.set_yscale("log")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p2 = out_dir / "benchmark_speedup.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → plot: {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark advisor vs FLAML — БЛОК 6 ТЗ.")
    parser.add_argument("--time-budget", type=int, default=60,
                        help="Time budget для FLAML, сек (default: 60).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Путь к KB CSV (full_results_v6.csv).")
    parser.add_argument("--no-synthetic-fallback", action="store_true",
                        help="Не использовать синтетический резерв при ошибке "
                             "загрузки датасета.")
    args = parser.parse_args()

    print("\n" + "█" * 72)
    print("  ВКР — ГЛАВА 4, БЛОК 6: BENCHMARK ADVISOR vs FLAML")
    print(f"  FLAML time_budget = {args.time_budget} s")
    print("█" * 72)

    # Загружаем KB
    kb_candidates = []
    if args.csv:
        kb_candidates.append(Path(args.csv))
    kb_candidates += [
        Path("results/full_results_v6.csv"),
        Path("results/full_results_v5.csv"),
    ]
    kb = None
    for p in kb_candidates:
        if p.exists():
            try:
                kb = KnowledgeBase.from_csv(p)
                print(f"  KB загружен из {p}")
                break
            except Exception as e:
                print(f"  [warn] {p}: {type(e).__name__}: {e}")
    if kb is None:
        kb = KnowledgeBase.from_defaults()
        print(f"  ⚠ Empirical KB не найден — используется literature-prior.")

    df = benchmark_against_flaml(
        kb=kb, time_budget=args.time_budget,
        use_synthetic_fallback=not args.no_synthetic_fallback)

    # Сохраняем
    csv_path = BENCHMARK_DIR / "benchmark_vs_flaml.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  → {csv_path}")

    # Графики
    plot_benchmark_results(df)

    # Сводка
    df_ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    if not df_ok.empty:
        print("\n  СВОДКА:")
        print(f"    Среднее rmse_ratio = {df_ok['rmse_ratio'].mean():.3f}  "
              f"(advisor / FLAML; <1 = advisor лучше)")
        print(f"    Медиана speedup    = ×{df_ok['speedup'].median():,.0f}")
        print(f"\n  Подробности — в {csv_path}")
    print()


if __name__ == "__main__":
    main()
