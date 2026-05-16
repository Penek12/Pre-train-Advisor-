"""
ВКР — Глава 3. Тестовый стенд (advisor v3) для апробации нелинейных
преобразований отклика. Pre-training рекомендации преобразований,
выученные из 27-датасетного бенчмарка (Глава 2).

ИЗМЕНЕНИЯ относительно v2:
  • БЛОК 5 ТЗ (Probabilistic advisor):
      Recommendation теперь содержит P10/P50/P90, prob_improvement
      (= P(ΔRMSE<0)) и prob_significant. Пользователь видит не точечный
      прогноз, а интервал — узкий = «надёжная» рекомендация, широкий =
      «рискованная». KnowledgeBase.from_csv пересчитывает quantile-агрегаты
      по бину. print_report / save_markdown_report показывают [P10, P90].

  • БЛОК 3 ТЗ (три стратегии): добавлен dataclass `Strategy`, который
      обобщает Recommendation на три подхода к асимметричному Y:
        – transform_Y          (преобразование Y; линейные модели)
        – glm                  (Gamma / Tweedie GLM, log-link)
        – specialized_loss     (XGBoost с reg:gamma / reg:tweedie)
      Функция `recommend_strategy(y, kb)` ранжирует все стратегии по
      прогнозу ΔRMSE и применимости — это и есть «advisor выбирает
      СТРАТЕГИЮ, а не преобразование».

  • Поведение для legacy-кода (старая `recommend()` с Recommendation)
    сохранено — мета-режим вызывается отдельной функцией.

Назначение
──────────
По заданному набору данных (X, y) и выбранной модели m возвращает
ранжированный список рекомендованных преобразований Y ещё ДО обучения
модели m. Рекомендация строится по двум источникам:

  (i)  knowledge base, извлечённый из full_results_v5.csv (Главы 2):
       для каждой пары (модель × γ₁-бин) — медианное ΔRMSE_test%,
       доля случаев со статистически значимым улучшением (DM_p<0.05),
       и количество подкрепляющих датасетов;

  (ii) фильтр применимости: исключает преобразования, не определённые
       на Y пользователя (например, log при наличии нулей без сдвига δ);

  (iii) теоретические корректировки (поправка Дуана при log; λ̂_MLE при
       Box-Cox; ширина 95% ДИ λ как мера устойчивости и т.д.).

МЕТОДОЛОГИЧЕСКОЕ ОГРАНИЧЕНИЕ
────────────────────────────
Преобразования отклика рекомендуются ТОЛЬКО для моделей класса линейной
регрессии (REGRESSION_MODELS = {linear, ridge, lasso}). Для древесных
моделей и MLP advisor возвращает единственную рекомендацию — «без
преобразования» — с пояснением причины (теория Box-Cox / Дуана требует
линейности E[g(Y)|X] и гомоскедастичности остатков; для деревьев и MLP
эти предпосылки не выполнены, а Йенсеново смещение при инверсии не
компенсируется поправкой Дуана). См. шапку chapter2_experiments_v5.py
для подробного обоснования.

Архитектура
───────────
    diagnose_target(y)            →  TargetDiagnostics
    KnowledgeBase.from_csv(path)  →  KnowledgeBase (эмпирический)
    KnowledgeBase.from_defaults() →  KnowledgeBase (literature prior)
    recommend(y, model_key, kb)   →  (diag, [Recommendation, ...])
    audit(X, y, model_key, kb)    →  (diag, recs, df_cv, actual_deltas)

Использование (Python API)
──────────────────────────
    from chapter3_advisor import (
        KnowledgeBase, diagnose_target, recommend, audit, print_report,
    )

    kb = KnowledgeBase.from_csv("results/full_results_v5.csv")
    diag, recs = recommend(y, model_key="linear", kb=kb, top_k=3,
                           verbose=True)
    # обучайте модель с recs[0].transform, см. Главу 2 apply_transform()

    # Опционально — эмпирическая проверка прогноза:
    diag, recs, df_cv, actual = audit(X, y, model_key="linear", kb=kb)

Запуск из консоли (демо на синтетике + Diamonds):
    python chapter3_advisor.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro
from scipy import stats
from scipy.stats import boxcox

# ── Импорт констант и pure-scipy функций (всегда нужны) ─────────────────────
# advisor_constants.py — лёгкий модуль без зависимостей от sklearn / xgboost /
# matplotlib / statsmodels. Грузится в любых окружениях, включая Streamlit
# Cloud, где могут отсутствовать тяжёлые пакеты для обучения моделей.
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
N_FOLDS         = 5
ALPHA           = 0.05      # уровень значимости

RESULTS_DIR = Path("results")
# НЕ создаём папку при импорте в облаке (read-only FS на некоторых платформах);
# создаём только если она нужна для записи — пусть это делает вызывающий код.

TRANSFORMS = ["none", "log", "sqrt", "asinh", "boxcox", "yeojohnson", "quantile"]
TR_LABEL = {
    "none":       "Без преобр.",
    "log":        "ln(y)",
    "sqrt":       "√y",
    "asinh":      "asinh(y)",
    "boxcox":     "Box-Cox",
    "yeojohnson": "Yeo-Johnson",
    "quantile":   "Quantile",
}
TR_CLASS = {
    "none":       "—",
    "log":        "степенное (фикс. λ→0)",
    "sqrt":       "степенное (фикс. λ=0.5)",
    "asinh":      "гиперболическое",
    "boxcox":     "степенное параметрическое (λ по MLE)",
    "yeojohnson": "степенное параметрическое (λ по MLE, R)",
    "quantile":   "ранговое непараметрическое",
}

MODELS = ["linear", "ridge", "lasso", "rf", "xgb", "mlp",
          "gamma_glm", "tweedie_glm", "xgb_gamma", "xgb_tweedie"]
MODEL_LABEL = {
    "linear": "МНК", "ridge": "Ridge", "lasso": "Lasso",
    "rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP",
    "gamma_glm":   "Gamma GLM",    "tweedie_glm": "Tweedie GLM",
    "xgb_gamma":   "XGB-Gamma",    "xgb_tweedie": "XGB-Tweedie",
}
MODEL_CLASS = {
    "linear":      "линейные",      "ridge":       "линейные",
    "lasso":       "линейные",
    "rf":          "древесные",     "xgb":         "древесные",
    "mlp":         "нейросетевые",
    "gamma_glm":   "glm",           "tweedie_glm": "glm",
    "xgb_gamma":   "древесные-glm-loss",
    "xgb_tweedie": "древесные-glm-loss",
}
REGRESSION_MODELS = frozenset({"linear", "ridge", "lasso"})


# ═════════════════════════════════════════════════════════════════════════════
# ПУРЕ-SCIPY ФУНКЦИИ (нужны для diagnose_target)
# Скопированы 1-в-1 из chapter2_experiments_v6.py
# ═════════════════════════════════════════════════════════════════════════════

def boxcox_fit(y_train):
    """Подбор λ по MLE + 95% доверительный интервал через LRT."""
    _, lam    = boxcox(y_train)
    ll_opt    = stats.boxcox_llf(lam, y_train)
    chi2_crit = stats.chi2.ppf(0.95, df=1)
    grid      = np.linspace(lam - 3.0, lam + 3.0, 3000)
    in_ci     = [l for l in grid
                 if 2.0 * (ll_opt - stats.boxcox_llf(l, y_train)) < chi2_crit]
    lo = min(in_ci) if in_ci else lam - 0.5
    hi = max(in_ci) if in_ci else lam + 0.5
    return float(lam), (float(lo), float(hi))


def lrt_boxcox(lam_null, lam_opt, y):
    """LRT: H₀: λ = lam_null."""
    ll_opt  = stats.boxcox_llf(lam_opt, y)
    ll_null = stats.boxcox_llf(lam_null, y)
    chi2    = float(-2.0 * (ll_null - ll_opt))
    p       = float(stats.chi2.sf(chi2, df=1))
    return chi2, p


# ── Lazy-import тяжёлой машинерии Главы 2 (только для audit()) ──────────────
# run_experiment_cv грузится по требованию — при первом вызове audit().
# В облачном режиме (без xgboost/sklearn/matplotlib) audit недоступен,
# но diagnose_target / recommend / recommend_strategy работают нормально.
_CHAPTER2_VERSION = None   # "v6" | "v5" | None — заполняется в _load_chapter2()
_run_experiment_cv = None  # будет заполнено при первом вызове audit()


def _load_chapter2():
    """Lazy-import chapter2_experiments_v6/v5. Вызывается только из audit()."""
    global _run_experiment_cv, _CHAPTER2_VERSION
    if _run_experiment_cv is not None:
        return _run_experiment_cv
    try:
        from chapter2_experiments_v6 import run_experiment_cv as _r
        _CHAPTER2_VERSION = "v6"
    except ImportError:
        try:
            from chapter2_experiments_v5 import run_experiment_cv as _r
            _CHAPTER2_VERSION = "v5"
        except ImportError as e:
            raise ImportError(
                "Эмпирический audit() требует chapter2_experiments_v6.py "
                "(или v5) со всеми зависимостями: sklearn, xgboost, "
                "statsmodels, matplotlib. Для облачного режима (только "
                "diagnose + recommend) audit не нужен — используйте UI без "
                f"галочки «Empirical audit». Причина: {e}")
    _run_experiment_cv = _r
    return _r


def is_audit_available() -> bool:
    """True, если chapter2 можно загрузить (для UI: показывать чекбокс или нет)."""
    try:
        _load_chapter2()
        return True
    except ImportError:
        return False

# Обратные карты «русский label → внутренний ключ»
LABEL_TO_TR    = {v: k for k, v in TR_LABEL.items()}
LABEL_TO_MODEL = {v: k for k, v in MODEL_LABEL.items()}

# ── Бины асимметрии (те же, что использовались в Главе 2) ────────────────────
GAMMA_EDGES  = [-np.inf, 0.5, 1.5, 3.0, np.inf]
GAMMA_BINS   = [
    "S — γ₁≤0.5 (квази-симметрично)",
    "M — 0.5<γ₁≤1.5 (умеренная асимметрия)",
    "H — 1.5<γ₁≤3 (сильная асимметрия)",
    "X — γ₁>3 (экстремальная асимметрия)",
]

ADVISOR_DIR = Path("advisor_output")
ADVISOR_DIR.mkdir(exist_ok=True)


def gamma_bin_of(g: float) -> str:
    """Возвращает строковую метку бина для значения γ₁."""
    for i in range(len(GAMMA_EDGES) - 1):
        if GAMMA_EDGES[i] < g <= GAMMA_EDGES[i + 1]:
            return GAMMA_BINS[i]
    return GAMMA_BINS[0]


# ═════════════════════════════════════════════════════════════════════════════
# 1. ДИАГНОСТИКА ОТКЛИКА Y (БЕЗ ОБУЧЕНИЯ МОДЕЛЕЙ)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TargetDiagnostics:
    """Все «дешёвые» статистики Y, не требующие обучения модели.

    Используется advisor-ом (BIN-KB и continuous meta-model) для
    рекомендации преобразования / стратегии. Все поля заполняются
    функцией diagnose_target().
    """
    n: int
    gamma1: float
    excess_kurt: float
    mean: float
    median: float
    std: float
    minv: float
    maxv: float
    zeros_pct: float
    neg_pct: float
    shapiro_p: float
    boxcox_lambda: Optional[float]
    boxcox_ci_lo: Optional[float]
    boxcox_ci_hi: Optional[float]
    lrt_log_chi2: Optional[float]    # H₀: λ = 0 (log)
    lrt_log_p: Optional[float]
    lrt_none_chi2: Optional[float]   # H₀: λ = 1 (без преобр.)
    lrt_none_p: Optional[float]
    gamma_bin: str
    suggested_shift: float           # δ для log/Box-Cox, если есть нули/отриц.
    # ── БЛОК 4 ТЗ: дополнительные фичи для meta-model ─────────────────────
    boxcox_ci_width: Optional[float] = None  # = ci_hi − ci_lo (мера устойчивости)
    log_n: Optional[float] = None             # log(n)
    mean_log_abs_Y: Optional[float] = None    # mean(log(|Y|+eps)) — масштабная фича

    def to_dict(self) -> dict:
        return asdict(self)

    def to_feature_vector(self) -> dict:
        """Плоский dict из 10 фич для meta-model (см. chapter3_meta_model.py)."""
        return {
            "gamma1":           float(self.gamma1),
            "excess_kurt":      float(self.excess_kurt),
            "log_n":            float(self.log_n) if self.log_n is not None
                                else float(np.log(max(self.n, 1))),
            "p_features":       float("nan"),  # заполняется снаружи
            "zeros_pct":        float(self.zeros_pct),
            "neg_pct":          float(self.neg_pct),
            "boxcox_lambda":    float(self.boxcox_lambda)
                                if self.boxcox_lambda is not None else float("nan"),
            "boxcox_ci_width":  float(self.boxcox_ci_width)
                                if self.boxcox_ci_width is not None else float("nan"),
            "shapiro_p":        float(self.shapiro_p)
                                if not np.isnan(self.shapiro_p) else float("nan"),
            "mean_log_abs_Y":   float(self.mean_log_abs_Y)
                                if self.mean_log_abs_Y is not None else float("nan"),
        }


def diagnose_target(y, shift: Optional[float] = None,
                    bc_sample_size: int = 10_000) -> TargetDiagnostics:
    """
    Полная диагностика отклика Y. Стоимость: O(n log n) + одна 1D-оптимизация
    Box-Cox; никаких обучений модели.

    Параметры
    ─────────
    y       : array-like, отклик
    shift   : δ для log/Box-Cox. Если None — автоматически выбирается так,
              чтобы min(y+δ) ≥ 1 при наличии нулей/отрицательных значений.
    bc_sample_size : сабсэмпл для Box-Cox MLE и LRT (для скорости при больших n).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 8:
        raise ValueError(f"n={n}: слишком мало для диагностики (нужно ≥ 8)")

    g1   = float(skew(y))
    kurt = float(kurtosis(y))   # excess kurtosis (нормальное → 0)

    minv = float(np.min(y)); maxv = float(np.max(y))
    zeros = int((y == 0).sum())
    negs  = int((y < 0).sum())

    # Авто-сдвиг для log/Box-Cox: чтобы min(y+δ) = 1
    if shift is None:
        shift = float(1.0 - minv) if minv <= 0 else 0.0

    # Shapiro–Wilk (на сэмпле, если n большое — иначе p-value становится
    # неинформативным «нулём» из-за power)
    try:
        sample = y if n <= 5000 else np.random.default_rng(RANDOM_STATE
                                        ).choice(y, 5000, replace=False)
        sh_p = float(shapiro(sample)[1])
    except Exception:
        sh_p = float("nan")

    # Box-Cox MLE + LRT (требует Y > 0 после сдвига)
    lam = lo = hi = chi2_log = p_log = chi2_none = p_none = None
    y_shift = y + shift
    if np.all(y_shift > 0):
        try:
            samp = y_shift if n <= bc_sample_size else np.random.default_rng(
                RANDOM_STATE).choice(y_shift, bc_sample_size, replace=False)
            lam, (lo, hi) = boxcox_fit(samp)
            chi2_log,  p_log  = lrt_boxcox(0.0, lam, samp)
            chi2_none, p_none = lrt_boxcox(1.0, lam, samp)
        except Exception:
            pass

    return TargetDiagnostics(
        n=n, gamma1=g1, excess_kurt=kurt,
        mean=float(y.mean()), median=float(np.median(y)), std=float(y.std()),
        minv=minv, maxv=maxv,
        zeros_pct=zeros / n * 100, neg_pct=negs / n * 100,
        shapiro_p=sh_p,
        boxcox_lambda=lam, boxcox_ci_lo=lo, boxcox_ci_hi=hi,
        lrt_log_chi2=chi2_log,   lrt_log_p=p_log,
        lrt_none_chi2=chi2_none, lrt_none_p=p_none,
        gamma_bin=gamma_bin_of(g1),
        suggested_shift=shift,
        # БЛОК 4 ТЗ: фичи для meta-model
        boxcox_ci_width=(float(hi - lo) if (lo is not None and hi is not None)
                         else None),
        log_n=float(np.log(max(n, 1))),
        mean_log_abs_Y=float(np.mean(np.log(np.abs(y) + 1e-9))),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. KNOWLEDGE BASE
# ═════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    База знаний: для каждой пары (model_key, γ₁_bin) — ранжированная таблица
    преобразований с агрегированными метриками: medi/mean ΔRMSE_test%,
    доля случаев со статистически значимым улучшением, n_evidence.

    Источники:
      • from_csv("full_results_v5.csv")  — эмпирический (предпочтительно);
      • from_defaults()                  — априорный (literature-informed
        prior; используется когда результаты Главы 2 ещё не получены).
    """

    REQUIRED_COLS = ("Dataset", "gamma1", "Model", "Transform",
                     "delta_RMSE_test_pct")

    def __init__(self, rules_df: pd.DataFrame, source: str = "unknown"):
        self._rules = rules_df.copy()
        self.source = source

    # ── Источник 1: реальные результаты Главы 2 ──────────────────────────────
    @classmethod
    def from_csv(cls, csv_path) -> "KnowledgeBase":
        df = pd.read_csv(csv_path)
        missing = [c for c in cls.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path}: отсутствуют колонки {missing}. Ожидается формат "
                "results/full_results_v5.csv (вывод Главы 2).")

        df = df.copy()
        df["Model_key"]     = df["Model"].map(LABEL_TO_MODEL)
        df["Transform_key"] = df["Transform"].map(LABEL_TO_TR)
        df["gamma_bin"]     = df["gamma1"].apply(gamma_bin_of)

        # Baseline (none) исключаем: ΔRMSE = 0 по построению
        df = df[df["Transform_key"] != "none"].dropna(subset=["Model_key"])

        rows = []
        for (mkey, gbin, tkey), sub in df.groupby(
                ["Model_key", "gamma_bin", "Transform_key"]):
            delta = sub["delta_RMSE_test_pct"].astype(float).values
            DM_p  = (sub["DM_p"].astype(float).values
                     if "DM_p" in sub.columns else np.full(len(sub), np.nan))
            # ── БЛОК 5 ТЗ: probabilistic-метрики ────────────────────────────
            # P10/P50/P90 — персентили распределения ΔRMSE по датасетам в бине.
            # prob_improvement = P(ΔRMSE < 0) — любое улучшение.
            # sig_better_rate  = P(значимое улучшение, DM_p<0.05 и ΔRMSE<0).
            if len(delta) > 0:
                p10 = float(np.nanpercentile(delta, 10))
                p50 = float(np.nanmedian(delta))
                p90 = float(np.nanpercentile(delta, 90))
                prob_improvement = float(np.mean(delta < 0))
            else:
                p10 = p50 = p90 = float("nan")
                prob_improvement = 0.0
            sig_better_rate = float(np.mean(
                (~np.isnan(DM_p)) & (DM_p < ALPHA) & (delta < 0)
            )) if len(delta) else 0.0
            rows.append({
                "Model_key": mkey,
                "gamma_bin": gbin,
                "Transform_key": tkey,
                # legacy-имена (для обратной совместимости с advisor v2)
                "median_delta_pct": p50,
                "mean_delta_pct":   float(np.nanmean(delta)) if len(delta)
                                     else float("nan"),
                "std_delta_pct":    float(np.nanstd(delta)) if len(delta)
                                     else float("nan"),
                "n_evidence":       int(len(delta)),
                "sig_better_rate":  sig_better_rate,
                # БЛОК 5 ТЗ
                "p10_delta_pct":    p10,
                "p50_delta_pct":    p50,   # = median (для ясности имени)
                "p90_delta_pct":    p90,
                "prob_improvement": prob_improvement,
            })
        rules = pd.DataFrame(rows)
        rules["rank"] = rules.groupby(
            ["Model_key", "gamma_bin"])["median_delta_pct"
        ].rank(method="min").astype(int)
        return cls(rules, source=str(csv_path))

    # ── Источник 2: априорный (literature prior) ─────────────────────────────
    @classmethod
    def from_defaults(cls) -> "KnowledgeBase":
        """
        Априорная таблица по литературе (Box & Cox 1964; Carroll & Ruppert 1988;
        Sakia 1992; Yeo & Johnson 2000; Hyndman & Athanasopoulos 2018).

        Используется ТОЛЬКО до получения эмпирических результатов Главы 2.
        Значения — порядковые оценки, не точные прогнозы.

        Замечание: prior формируется ТОЛЬКО для моделей из REGRESSION_MODELS
        (МНК, Ridge, Lasso). Для древесных моделей и MLP преобразования
        отклика методологически не применяются, поэтому в KB для них нет
        правил (см. recommend(): для таких моделей выдаётся специальная
        рекомендация «без преобразования»).
        """
        # ΔRMSE (медиана) по бинам для линейных моделей и Box-Cox
        # (наиболее изученный случай). Остальные преобразования
        # шкалируются от этой опорной точки.
        prior_linear = {
            #  bin   : [log, sqrt, asinh, boxcox, yeojohnson, quantile]
            GAMMA_BINS[0]: [+5,   +2,   +1,   -1,   -1,   +3],  # симметрия
            GAMMA_BINS[1]: [-8,   -5,   -7,  -10,  -10,   -5],  # умеренная
            GAMMA_BINS[2]: [-20, -12,  -18,  -25,  -23,  -15],  # сильная
            GAMMA_BINS[3]: [-35, -18,  -30,  -40,  -38,  -25],  # экстремальная
        }
        sig_rate_linear = {GAMMA_BINS[0]: 0.05, GAMMA_BINS[1]: 0.40,
                           GAMMA_BINS[2]: 0.75, GAMMA_BINS[3]: 0.92}

        TR_ORDER = ["log", "sqrt", "asinh", "boxcox", "yeojohnson", "quantile"]
        rows = []
        for gbin, deltas in prior_linear.items():
            # Только регрессионные модели — для деревьев и MLP правил нет
            # по построению (см. docstring).
            for m in REGRESSION_MODELS:
                for tr, d in zip(TR_ORDER, deltas):
                    rows.append((m, gbin, tr, d, sig_rate_linear[gbin]))

        df = pd.DataFrame(rows, columns=[
            "Model_key", "gamma_bin", "Transform_key",
            "median_delta_pct", "sig_better_rate"])
        df["mean_delta_pct"] = df["median_delta_pct"]
        df["std_delta_pct"]  = 0.0
        df["n_evidence"]     = 0          # маркер: prior, не эмпирика
        # БЛОК 5 ТЗ: априорные оценки P10/P90 — ±50% от медианы (грубо).
        # Поскольку prior получен из литературы (точечные оценки), реальный
        # разброс неизвестен; используем эвристику для совместимости интерфейса.
        df["p10_delta_pct"]    = df["median_delta_pct"] - 10.0
        df["p50_delta_pct"]    = df["median_delta_pct"]
        df["p90_delta_pct"]    = df["median_delta_pct"] + 10.0
        df["prob_improvement"] = (df["median_delta_pct"] < 0).astype(float) \
                                  * 0.7 + 0.15   # грубая эвристика: 0.85 / 0.15
        df["rank"] = df.groupby(
            ["Model_key", "gamma_bin"])["median_delta_pct"
        ].rank(method="min").astype(int)
        return cls(df, source="literature-prior (априорный, regression-only)")

    # ── Запросы ──────────────────────────────────────────────────────────────
    def get(self, model_key: str, gamma1: float,
            top_k: Optional[int] = None) -> pd.DataFrame:
        gbin = gamma_bin_of(gamma1)
        sub = self._rules[
            (self._rules["Model_key"] == model_key) &
            (self._rules["gamma_bin"] == gbin)
        ].sort_values("median_delta_pct").reset_index(drop=True)
        return sub.head(top_k) if top_k else sub

    def coverage(self) -> pd.DataFrame:
        """Сколько эмпирических точек в каждой клетке."""
        return self._rules.pivot_table(
            index="Model_key", columns="gamma_bin",
            values="n_evidence", aggfunc="max", fill_value=0)

    def to_csv(self, path) -> None:
        self._rules.to_csv(path, index=False, encoding="utf-8-sig")

    def summary(self) -> None:
        n_models = self._rules["Model_key"].nunique()
        n_bins   = self._rules["gamma_bin"].nunique()
        n_rules  = len(self._rules)
        n_emp    = int((self._rules["n_evidence"] > 0).sum())
        print(f"  KnowledgeBase: {self.source}")
        print(f"    моделей={n_models}, γ₁-бинов={n_bins}, "
              f"правил={n_rules} (эмпирических: {n_emp})")


# ═════════════════════════════════════════════════════════════════════════════
# 3. ФИЛЬТР ПРИМЕНИМОСТИ
# ═════════════════════════════════════════════════════════════════════════════

def applicability(transform: str, diag: TargetDiagnostics) -> tuple:
    """Возвращает (bool_применимо, str_причина_если_нет)."""
    if transform in ("none", "asinh", "yeojohnson", "quantile"):
        return True, ""

    min_after_shift = diag.minv + diag.suggested_shift

    if transform in ("log", "boxcox"):
        if min_after_shift <= 0:
            return False, f"требует Y>0; min(Y+δ)={min_after_shift:.3g}"
        if diag.zeros_pct > 0 and diag.suggested_shift == 0:
            return False, (f"в Y нули ({diag.zeros_pct:.1f}%), нужен δ>0; "
                           f"используйте suggested_shift={diag.suggested_shift}")
        return True, ""

    if transform == "sqrt":
        if min_after_shift < 0:
            return False, f"требует Y≥0; min(Y+δ)={min_after_shift:.3g}"
        return True, ""

    return True, ""


# ═════════════════════════════════════════════════════════════════════════════
# 4. РЕКОМЕНДАЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Recommendation:
    rank: int
    transform: str
    transform_label: str
    transform_class: str
    predicted_delta_pct: float       # = delta_p50 (медиана; для обратной совместимости)
    sig_better_rate: float           # = prob_significant
    n_evidence: int
    applicable: bool
    rationale: str
    warnings: List[str] = field(default_factory=list)
    # ── БЛОК 5 ТЗ: probabilistic advisor ────────────────────────────────────
    # P10/P50/P90 — персентили распределения ΔRMSE по датасетам в бине.
    # Узкий интервал [P10, P90] = «надёжная» рекомендация; широкий = «рискованная».
    delta_p10: float = float("nan")            # 10-й персентиль ΔRMSE
    delta_p50: float = float("nan")            # медиана (= predicted_delta_pct)
    delta_p90: float = float("nan")            # 90-й персентиль ΔRMSE
    prob_improvement: float = float("nan")     # P(ΔRMSE < 0) — любое улучшение
    prob_significant: float = float("nan")     # P(значимое улучшение, DM_p<0.05)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 3 ТЗ: Strategy dataclass — три пути обработки асимметричного Y
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy обобщает Recommendation на три подхода к Y:
#   1) transform_Y       — преобразование Y (g(Y) = Xβ + ε) для линейных моделей.
#                          Преимущество: интерпретируемые коэффициенты, λ̂_MLE.
#                          Недостаток: Йенсеново смещение при инверсии,
#                                       требует поправки Дуана / smearing.
#   2) glm               — обобщённая линейная модель с подходящим семейством:
#                          Gamma (Y > 0) или Tweedie (Y ≥ 0, нули допустимы).
#                          Y остаётся в исходной шкале, log-link учитывает
#                          мультипликативный характер; Йенсеновой проблемы нет.
#   3) specialized_loss  — XGBoost с alt-loss (reg:gamma / reg:tweedie).
#                          Сохраняет преимущества деревьев (взаимодействия,
#                          нелинейность) + Gamma/Tweedie likelihood вместо MSE.
#                          Y в исходной шкале; не требует λ̂_MLE.
#
# advisor выбирает между этими тремя на основе диагностики Y (γ₁, нули,
# отрицательные значения, n, p_features).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Strategy:
    """Одна из трёх стратегий обработки асимметричного Y."""
    rank: int                       # 1 = лучшая по прогнозу
    model_key: str                  # 'linear' | 'ridge' | 'lasso' | 'gamma_glm' |
                                    # 'tweedie_glm' | 'xgb_gamma' | 'xgb_tweedie'
    model_label: str                # человекочитаемое имя
    transform: str                  # 'none' для GLM и xgb_*; иначе — какое g(Y)
    transform_label: str
    approach: str                   # 'transform_Y' | 'glm' | 'specialized_loss'
    predicted_delta_pct: float      # медиана прогноза ΔRMSE_test%
    delta_p10: float                # 10-й персентиль (нижняя граница диапазона)
    delta_p90: float                # 90-й персентиль
    prob_improvement: float         # P(ΔRMSE < 0)
    prob_significant: float         # P(значимое улучшение)
    n_evidence: int                 # сколько датасетов подкрепляет
    applicable: bool                # применима ли стратегия к данному Y
    rationale: str                  # объяснение для пользователя
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def recommend(y, model_key: str, kb: Optional[KnowledgeBase] = None,
              shift: Optional[float] = None, top_k: int = 3,
              verbose: bool = False) -> tuple:
    """
    Pre-training рекомендация. Возвращает (diag, [Recommendation, ...]).

    Параметры
    ─────────
    y          : array-like, отклик
    model_key  : str ∈ {'linear','ridge','lasso','rf','xgb','mlp'}
    kb         : KnowledgeBase; если None — берётся literature prior
    shift      : δ для log/Box-Cox; если None — авто
    top_k      : сколько рекомендаций вернуть
    verbose    : печать отчёта в stdout

    Методологическое ограничение
    ────────────────────────────
    Если model_key ∉ REGRESSION_MODELS (т.е. это RF, XGBoost или MLP),
    advisor выдаёт ЕДИНСТВЕННУЮ рекомендацию — «без преобразования» —
    с пояснением: теория Box-Cox / Дуана выведена для линейной модели
    с гомоскедастичными остатками, а перенос её на деревья и MLP не
    обоснован. См. шапку chapter2_experiments_v5.py.
    """
    if model_key not in MODELS:
        raise ValueError(f"model_key должен быть из {MODELS}, передано: {model_key}")

    diag = diagnose_target(y, shift=shift)

    # ── Спец-ветка: нерегрессионные модели ───────────────────────────────────
    # Для деревьев и MLP преобразования отклика не применяются методологически.
    # Возвращаем одну рекомендацию — 'none' — с пояснением.
    if model_key not in REGRESSION_MODELS:
        rationale = (
            f"Модель «{MODEL_LABEL[model_key]}» относится к классу "
            f"«{MODEL_CLASS[model_key]}». Нелинейные преобразования отклика "
            f"и поправка Дуана применяются только к линейным регрессиям "
            f"(МНК, Ridge, Lasso): теория Box & Cox (1964) и Дуана (1983) "
            f"исходит из модели g(Y)=Xβ+ε с гомоскедастичными остатками. "
            f"Деревья — непараметрические и сами улавливают нелинейность "
            f"E[Y|X]; MLP — универсальный аппроксиматор. Преобразование Y "
            f"здесь создаёт Йенсеново смещение при инверсии, которое "
            f"не компенсируется поправкой Дуана. Рекомендация: обучать "
            f"модель на Y в исходной шкале (transform='none')."
        )
        rec_none = Recommendation(
            rank=1,
            transform="none",
            transform_label=TR_LABEL["none"],
            transform_class=TR_CLASS["none"],
            predicted_delta_pct=0.0,
            sig_better_rate=0.0,
            n_evidence=0,
            applicable=True,
            rationale=rationale,
            warnings=[
                "Преобразования Y к этому классу моделей не применяются.",
                f"Если асимметрия γ₁={diag.gamma1:+.2f} велика и нужно "
                "масштабное преобразование — рассмотрите линейную регрессию "
                "(МНК / Ridge / Lasso) с подходящим преобразованием Y; "
                "advisor для линейных моделей выдаст содержательный список.",
            ],
            # БЛОК 5 ТЗ
            delta_p10=0.0, delta_p50=0.0, delta_p90=0.0,
            prob_improvement=0.0, prob_significant=0.0,
        )
        recs = [rec_none]
        if verbose:
            print_report(diag, recs, model_key)
        return diag, recs

    # ── Стандартная ветка: регрессионные модели ──────────────────────────────
    if kb is None:
        kb = KnowledgeBase.from_defaults()

    rules = kb.get(model_key, diag.gamma1)
    if rules.empty:
        # Если в KB нет данных для этого бина — fallback на априорный
        prior = KnowledgeBase.from_defaults()
        rules = prior.get(model_key, diag.gamma1)
        kb_source_note = f"(fallback prior, в основном KB нет данных для бина)"
    else:
        kb_source_note = ""

    recs: List[Recommendation] = []
    for _, r in rules.iterrows():
        tr = r["Transform_key"]
        ok, reason = applicability(tr, diag)
        warns: List[str] = []
        if not ok:
            warns.append(f"Не применимо: {reason}")
        # ── Предупреждение о Йенсеновом смещении при наивной инверсии ──
        # Для всех нелинейных монотонных g неравенство Йенсена даёт
        # E[g⁻¹(ĝ + ε)] ≠ g⁻¹(ĝ). Реализация: см. general_smearing
        # в chapter2_experiments_v5.py (use_smearing=True в run_experiment_cv).
        # Поправка не нужна для quantile (ECDF-based) и none (тривиально).
        SMEAR_NEEDED = {"log", "sqrt", "asinh", "boxcox", "yeojohnson"}
        if tr in SMEAR_NEEDED:
            try:
                y_arr = np.asarray(y, dtype=float)
                y_pos = y_arr + diag.suggested_shift
                if tr in ("log", "boxcox") and (y_pos > 0).all():
                    sigma_g = float(np.std(np.log(y_pos)))
                    bias_kind = "exp(σ²_log/2)"
                    bias_val = float(np.exp(sigma_g ** 2 / 2.0))
                elif tr == "sqrt" and (y_pos >= 0).all():
                    sigma_g = float(np.std(np.sqrt(y_pos)))
                    bias_kind = "+σ²_sqrt"
                    bias_val = sigma_g ** 2
                elif tr == "asinh":
                    sigma_g = float(np.std(np.arcsinh(y_arr + diag.suggested_shift)))
                    bias_kind = "cosh(σ)"
                    bias_val = float(np.cosh(sigma_g))
                elif tr == "yeojohnson":
                    from sklearn.preprocessing import PowerTransformer
                    pt = PowerTransformer(method="yeo-johnson", standardize=False)
                    y_yj = pt.fit_transform(y_arr.reshape(-1, 1)).ravel()
                    sigma_g = float(np.std(y_yj))
                    lam_yj = float(pt.lambdas_[0])
                    bias_kind = f"Yeo-Johnson (λ̂={lam_yj:.2f})"
                    # Грубая оценка масштаба смещения через σ
                    bias_val = float(np.exp(sigma_g ** 2 / 2.0)) \
                        if abs(lam_yj) < 0.5 else 1.0 + sigma_g ** 2 / 2.0
                else:
                    sigma_g, bias_kind, bias_val = float("nan"), "—", float("nan")
                warns.append(
                    f"Неравенство Йенсена: наивная инверсия g⁻¹(ĝ) "
                    f"даёт смещение ~{bias_kind} ≈ {bias_val:.3f} "
                    f"(σ_g≈{sigma_g:.3f}). При обучении используйте "
                    f"`run_experiment_cv(..., use_smearing=True)` — это включает "
                    f"обобщённую поправку Дуана (general_smearing) для всех "
                    f"монотонных преобразований."
                )
            except Exception:
                warns.append(
                    "Неравенство Йенсена: наивная инверсия g⁻¹(ĝ) смещена. "
                    "Используйте `use_smearing=True` в run_experiment_cv."
                )
        if tr == "boxcox" and diag.boxcox_lambda is not None:
            warns.append(f"λ̂_MLE = {diag.boxcox_lambda:.3f}, "
                         f"95%ДИ [{diag.boxcox_ci_lo:.3f}; {diag.boxcox_ci_hi:.3f}]")
            if (diag.boxcox_ci_hi - diag.boxcox_ci_lo) > 1.5:
                warns.append("Широкий ДИ λ → результат может быть неустойчивым")
        if tr == "quantile":
            warns.append("Стирает информацию о расстояниях (только порядок); "
                         "не подходит, если важна интерпретация шкалы")

        n_ev = int(r["n_evidence"])
        ev_str = f"по данным {n_ev} датасетов" if n_ev > 0 else "по prior'у"
        rationale = (
            f"γ₁={diag.gamma1:+.2f} (бин {diag.gamma_bin}); модель "
            f"{MODEL_LABEL[model_key]}: преобразование «{TR_LABEL[tr]}» "
            f"{ev_str} даёт медианное ΔRMSE_test = {r['median_delta_pct']:+.1f}%, "
            f"P(значимо лучше) = {r['sig_better_rate']*100:.0f}%."
        )
        if kb_source_note:
            rationale += " " + kb_source_note

        recs.append(Recommendation(
            rank=0,  # переприсвоим после сортировки
            transform=tr,
            transform_label=TR_LABEL[tr],
            transform_class=TR_CLASS[tr],
            predicted_delta_pct=float(r["median_delta_pct"]),
            sig_better_rate=float(r["sig_better_rate"]),
            n_evidence=n_ev,
            applicable=ok,
            rationale=rationale,
            warnings=warns,
            # ── БЛОК 5 ТЗ: probabilistic-поля ──
            delta_p10=float(r.get("p10_delta_pct", float("nan"))),
            delta_p50=float(r.get("p50_delta_pct", r["median_delta_pct"])),
            delta_p90=float(r.get("p90_delta_pct", float("nan"))),
            prob_improvement=float(r.get("prob_improvement", float("nan"))),
            prob_significant=float(r["sig_better_rate"]),
        ))

    # Сортировка: применимые → по возрастанию predicted_delta_pct (меньше = лучше)
    recs.sort(key=lambda r: (not r.applicable, r.predicted_delta_pct))
    for i, r in enumerate(recs, 1):
        r.rank = i
    recs = recs[:top_k]

    if verbose:
        print_report(diag, recs, model_key)

    return diag, recs


# ═════════════════════════════════════════════════════════════════════════════
# 4b. РЕКОМЕНДАЦИЯ СТРАТЕГИИ (БЛОК 3 ТЗ) — три пути обработки Y
# ═════════════════════════════════════════════════════════════════════════════

# Какие model_key соответствуют каким approach-ам:
APPROACH_BY_MODEL = {
    "linear":       "transform_Y",
    "ridge":        "transform_Y",
    "lasso":        "transform_Y",
    "gamma_glm":    "glm",
    "tweedie_glm":  "glm",
    "xgb_gamma":    "specialized_loss",
    "xgb_tweedie":  "specialized_loss",
    # Базовый XGBoost и RF/MLP без alt-loss — не «стратегии», а отдельный класс
    # (для справки) — их Strategy.approach = 'baseline_treebased' / 'neural'
    "xgb":          "baseline_treebased",
    "rf":           "baseline_treebased",
    "mlp":          "neural",
}

APPROACH_LABEL = {
    "transform_Y":       "преобразование Y (линейная модель)",
    "glm":               "GLM с подходящим распределением (log-link)",
    "specialized_loss":  "XGBoost с alt-loss (reg:gamma / reg:tweedie)",
    "baseline_treebased": "дерево без alt-loss (контроль)",
    "neural":            "нейросеть (контроль)",
}


def _glm_applicability(model_key: str, diag: TargetDiagnostics) -> tuple:
    """Применимость GLM/specialized-loss моделей по диапазону Y.

    Возвращает (bool_применимо, str_причина_если_нет).
    """
    # Y > 0 строго: Gamma GLM, XGB-Gamma
    if model_key in ("gamma_glm", "xgb_gamma"):
        if diag.minv <= 0:
            return False, (f"требует Y > 0; min(Y)={diag.minv:.3g}. "
                           f"Для Y с нулями/отриц. используйте Tweedie-вариант.")
        return True, ""
    # Y ≥ 0 (нули допустимы, отрицательные — нет): Tweedie GLM, XGB-Tweedie
    if model_key in ("tweedie_glm", "xgb_tweedie"):
        if diag.minv < 0:
            return False, (f"требует Y ≥ 0 (с var_power=1.5); "
                           f"min(Y)={diag.minv:.3g}. Для Y с отрицательными "
                           f"значениями стратегия неприменима.")
        return True, ""
    return True, ""


def recommend_strategy(y, kb: Optional[KnowledgeBase] = None,
                       shift: Optional[float] = None,
                       top_k: int = 5,
                       include_baselines: bool = False,
                       verbose: bool = False) -> tuple:
    """BLOCK 3 ТЗ: pre-training рекомендация СТРАТЕГИИ (а не преобразования).

    Объединяет три подхода в единый ранжированный список:
      • transform_Y    — линейная регрессия с g(Y);
      • glm            — Gamma / Tweedie GLM;
      • specialized_loss — XGBoost с reg:gamma / reg:tweedie.

    Возвращает (diag, [Strategy, ...]).

    Параметры
    ─────────
    y                : array-like
    kb               : KnowledgeBase; если None — берётся literature prior
                       (для transform_Y) + грубые эвристики для GLM/spec-loss
    shift            : δ для log/Box-Cox; авто, если None
    top_k            : сколько стратегий вернуть в финальном списке
    include_baselines: если True — добавляет в список xgb/rf/mlp как контроль
    verbose          : печать отчёта

    Замечание о KB для GLM/spec-loss
    ────────────────────────────────
    Если в KB есть прогноз для (gamma_glm, none), (xgb_gamma, none) и т.д. —
    используется он. Иначе делается грубая эвристика «GLM ≈ выигрыш как у log
    для линейной модели в этом бине», а для spec-loss — как у boxcox.
    """
    if kb is None:
        kb = KnowledgeBase.from_defaults()

    diag = diagnose_target(y, shift=shift)

    strategies: List[Strategy] = []

    # ── A. transform_Y: рекомендации для линейных моделей ────────────────────
    # Берём ridge как репрезентативную линейную модель (можно поменять на linear).
    # Для каждого преобразования из KB → Strategy.
    for lin_model in ("linear", "ridge", "lasso"):
        rules = kb.get(lin_model, diag.gamma1)
        if rules.empty:
            continue
        for _, r in rules.iterrows():
            tr = r["Transform_key"]
            ok, reason = applicability(tr, diag)
            warns = []
            if not ok:
                warns.append(f"Не применимо: {reason}")
            rationale = (
                f"γ₁={diag.gamma1:+.2f} ({diag.gamma_bin}). "
                f"Подход «transform_Y»: модель «{MODEL_LABEL[lin_model]}» "
                f"+ преобразование «{TR_LABEL[tr]}». "
                f"Медиана прогноза ΔRMSE = {r['median_delta_pct']:+.1f}% "
                f"по {int(r['n_evidence'])} датасетам бина."
            )
            strategies.append(Strategy(
                rank=0,
                model_key=lin_model, model_label=MODEL_LABEL[lin_model],
                transform=tr, transform_label=TR_LABEL[tr],
                approach="transform_Y",
                predicted_delta_pct=float(r["median_delta_pct"]),
                delta_p10=float(r.get("p10_delta_pct", float("nan"))),
                delta_p90=float(r.get("p90_delta_pct", float("nan"))),
                prob_improvement=float(r.get("prob_improvement", float("nan"))),
                prob_significant=float(r["sig_better_rate"]),
                n_evidence=int(r["n_evidence"]),
                applicable=ok,
                rationale=rationale,
                warnings=warns,
            ))

    # ── B. glm: Gamma / Tweedie GLM ─────────────────────────────────────────
    # Подход не требует преобразования Y; transform='none'.
    # Прогноз ΔRMSE — из KB для (glm_model, none) если есть; иначе эвристика.
    for glm_model in ("gamma_glm", "tweedie_glm"):
        ok, reason = _glm_applicability(glm_model, diag)
        warns = []
        if not ok:
            warns.append(f"Не применимо: {reason}")

        # Пытаемся найти эмпирический прогноз в KB
        rules_glm = kb._rules[
            (kb._rules["Model_key"] == glm_model) &
            (kb._rules["gamma_bin"] == diag.gamma_bin) &
            (kb._rules["Transform_key"] == "none")
        ]
        if not rules_glm.empty:
            r_glm = rules_glm.iloc[0]
            predicted = float(r_glm["median_delta_pct"])
            p10 = float(r_glm.get("p10_delta_pct", predicted - 10))
            p90 = float(r_glm.get("p90_delta_pct", predicted + 10))
            prob_imp = float(r_glm.get("prob_improvement", float("nan")))
            prob_sig = float(r_glm["sig_better_rate"])
            n_ev = int(r_glm["n_evidence"])
            evidence_note = f"по {n_ev} датасетам"
        else:
            # Эвристика: GLM с log-link даёт приблизительно тот же выигрыш,
            # что и линейная модель + log в данном бине. Это нижняя оценка
            # (GLM, как правило, делает чуть лучше из-за корректной spec.).
            rules_log = kb._rules[
                (kb._rules["Model_key"] == "linear") &
                (kb._rules["gamma_bin"] == diag.gamma_bin) &
                (kb._rules["Transform_key"] == "log")
            ]
            if not rules_log.empty:
                r_log = rules_log.iloc[0]
                predicted = float(r_log["median_delta_pct"])
                p10 = float(r_log.get("p10_delta_pct", predicted - 10))
                p90 = float(r_log.get("p90_delta_pct", predicted + 10))
                prob_imp = float(r_log.get("prob_improvement", float("nan")))
                prob_sig = float(r_log["sig_better_rate"])
                n_ev = 0
                evidence_note = "(эвристика: проксируется linear+log в этом бине)"
            else:
                predicted = 0.0
                p10 = p90 = float("nan")
                prob_imp = float("nan")
                prob_sig = 0.0
                n_ev = 0
                evidence_note = "(нет данных)"

        rationale = (
            f"γ₁={diag.gamma1:+.2f} ({diag.gamma_bin}). "
            f"Подход «glm»: {MODEL_LABEL[glm_model]} с log-link сохраняет "
            f"Y в исходной шкале, не требует поправки Дуана. "
            f"Прогноз ΔRMSE ≈ {predicted:+.1f}% {evidence_note}."
        )
        warns.append("GLM использует IRLS; при экстремальной мультиколлинеарности "
                     "возможны проблемы сходимости — обёртка fallback'ит на mean.")
        strategies.append(Strategy(
            rank=0,
            model_key=glm_model, model_label=MODEL_LABEL[glm_model],
            transform="none", transform_label="без преобр.",
            approach="glm",
            predicted_delta_pct=predicted,
            delta_p10=p10, delta_p90=p90,
            prob_improvement=prob_imp, prob_significant=prob_sig,
            n_evidence=n_ev, applicable=ok,
            rationale=rationale, warnings=warns,
        ))

    # ── C. specialized_loss: XGBoost с alt-objective ─────────────────────────
    for xgb_model in ("xgb_gamma", "xgb_tweedie"):
        ok, reason = _glm_applicability(xgb_model, diag)
        warns = []
        if not ok:
            warns.append(f"Не применимо: {reason}")

        rules_xgb = kb._rules[
            (kb._rules["Model_key"] == xgb_model) &
            (kb._rules["gamma_bin"] == diag.gamma_bin) &
            (kb._rules["Transform_key"] == "none")
        ]
        if not rules_xgb.empty:
            r_xgb = rules_xgb.iloc[0]
            predicted = float(r_xgb["median_delta_pct"])
            p10 = float(r_xgb.get("p10_delta_pct", predicted - 10))
            p90 = float(r_xgb.get("p90_delta_pct", predicted + 10))
            prob_imp = float(r_xgb.get("prob_improvement", float("nan")))
            prob_sig = float(r_xgb["sig_better_rate"])
            n_ev = int(r_xgb["n_evidence"])
            evidence_note = f"по {n_ev} датасетам"
        else:
            # Эвристика: spec.loss даёт приблизительно тот же выигрыш на
            # асимметричном Y, что и Box-Cox у линейных моделей.
            rules_bc = kb._rules[
                (kb._rules["Model_key"] == "linear") &
                (kb._rules["gamma_bin"] == diag.gamma_bin) &
                (kb._rules["Transform_key"] == "boxcox")
            ]
            if not rules_bc.empty:
                r_bc = rules_bc.iloc[0]
                # XGBoost обычно даёт + ещё немного из-за деревьев и взаимодействий
                predicted = float(r_bc["median_delta_pct"]) * 0.8  # консервативно
                p10 = float(r_bc.get("p10_delta_pct", predicted - 10))
                p90 = float(r_bc.get("p90_delta_pct", predicted + 10))
                prob_imp = float(r_bc.get("prob_improvement", float("nan")))
                prob_sig = float(r_bc["sig_better_rate"]) * 0.7  # консервативно
                n_ev = 0
                evidence_note = "(эвристика: проксируется linear+boxcox × 0.8)"
            else:
                predicted = 0.0
                p10 = p90 = float("nan")
                prob_imp = float("nan")
                prob_sig = 0.0
                n_ev = 0
                evidence_note = "(нет данных)"

        rationale = (
            f"γ₁={diag.gamma1:+.2f} ({diag.gamma_bin}). "
            f"Подход «specialized_loss»: {MODEL_LABEL[xgb_model]} с "
            f"loss = reg:{xgb_model.split('_')[1]}. "
            f"Преимущество: alt-loss + взаимодействия признаков "
            f"(в отличие от GLM с линейным η). "
            f"Прогноз ΔRMSE ≈ {predicted:+.1f}% {evidence_note}."
        )
        strategies.append(Strategy(
            rank=0,
            model_key=xgb_model, model_label=MODEL_LABEL[xgb_model],
            transform="none", transform_label="без преобр.",
            approach="specialized_loss",
            predicted_delta_pct=predicted,
            delta_p10=p10, delta_p90=p90,
            prob_improvement=prob_imp, prob_significant=prob_sig,
            n_evidence=n_ev, applicable=ok,
            rationale=rationale, warnings=warns,
        ))

    # ── D. (опц.) baseline_treebased и neural — для сравнения ────────────────
    if include_baselines:
        for base_model in ("xgb", "rf", "mlp"):
            approach = APPROACH_BY_MODEL[base_model]
            rationale = (
                f"Контроль: {MODEL_LABEL[base_model]} без alt-loss и без "
                f"преобразования Y — baseline для сравнения с тремя стратегиями."
            )
            strategies.append(Strategy(
                rank=0,
                model_key=base_model, model_label=MODEL_LABEL[base_model],
                transform="none", transform_label="без преобр.",
                approach=approach,
                predicted_delta_pct=0.0,
                delta_p10=0.0, delta_p90=0.0,
                prob_improvement=0.0, prob_significant=0.0,
                n_evidence=0, applicable=True,
                rationale=rationale, warnings=[],
            ))

    # ── Сортировка: применимые → по возрастанию predicted_delta_pct ──────────
    strategies.sort(key=lambda s: (not s.applicable, s.predicted_delta_pct))
    for i, s in enumerate(strategies, 1):
        s.rank = i
    strategies = strategies[:top_k]

    if verbose:
        print_strategy_report(diag, strategies)

    return diag, strategies


def print_strategy_report(diag: TargetDiagnostics,
                          strategies: List[Strategy]) -> None:
    """Печать отчёта по стратегиям (БЛОК 3 ТЗ)."""
    print("\n" + "═" * 78)
    print("  ОТЧЁТ ADVISOR-А — РЕКОМЕНДАЦИЯ СТРАТЕГИИ (3 пути обработки Y)")
    print("═" * 78)

    print(f"\n  Диагностика отклика Y:  γ₁={diag.gamma1:+.3f} ({diag.gamma_bin})")
    print(f"     n={diag.n:,}, min={diag.minv:.3g}, max={diag.maxv:.3g}, "
          f"zeros={diag.zeros_pct:.1f}%, neg={diag.neg_pct:.1f}%")

    print(f"\n  Топ-{len(strategies)} стратегий (ранжированы по медиане прогноза ΔRMSE):")
    print("  " + "─" * 76)
    for s in strategies:
        flag = "✓" if s.applicable else "✗"
        approach_lbl = APPROACH_LABEL.get(s.approach, s.approach)
        if (not np.isnan(s.delta_p10)) and (not np.isnan(s.delta_p90)):
            range_str = f"[{s.delta_p10:+5.1f}%, {s.delta_p90:+5.1f}%]"
        else:
            range_str = "[—, —]"
        print(f"\n    [#{s.rank}] {flag}  {s.model_label}  +  "
              f"{s.transform_label}")
        print(f"         подход: {approach_lbl}")
        print(f"         медиана ΔRMSE = {s.predicted_delta_pct:+6.1f}%,  "
              f"диапазон {range_str}")
        if not np.isnan(s.prob_improvement):
            print(f"         P(улучш.) = {s.prob_improvement*100:3.0f}%   "
                  f"P(знач.) = {s.prob_significant*100:3.0f}%   "
                  f"n_evidence = {s.n_evidence}")
        print(f"         {s.rationale}")
        for w in s.warnings:
            print(f"         ⚠ {w}")
    print("\n  Подходы (для справки):")
    for ap_key, ap_lbl in APPROACH_LABEL.items():
        print(f"    • {ap_key:<22} — {ap_lbl}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# 4c. (исходный recommend для одной модели — продолжается ниже)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# 5. AUDIT — эмпирическая проверка рекомендации на данных пользователя
# ═════════════════════════════════════════════════════════════════════════════

def audit(X, y, model_key: str, kb: Optional[KnowledgeBase] = None,
          transforms_to_test: Optional[List[str]] = None,
          shift: Optional[float] = None, verbose: bool = True) -> tuple:
    """
    Запускает run_experiment_cv (5-fold CV + DM на hold-out) для подмножества
    преобразований и сравнивает фактическое ΔRMSE с прогнозом advisor-а.

    Возвращает (diag, recs, df_cv, actual_deltas_dict).

    Методологическое ограничение
    ────────────────────────────
    Для model_key ∉ REGRESSION_MODELS оценивается только baseline 'none' —
    преобразования к древесным моделям и MLP не применяются (см. recommend).
    """
    diag, recs = recommend(y, model_key, kb=kb, shift=shift, top_k=3)

    # Для нерегрессионных моделей run_experiment_cv по построению вернёт
    # только строку 'none'. Принудительно ограничиваем список тестируемых
    # преобразований.
    if model_key not in REGRESSION_MODELS:
        transforms_to_test = ["none"]
    elif transforms_to_test is None:
        transforms_to_test = [r.transform for r in recs if r.applicable]
    if "none" not in transforms_to_test:
        transforms_to_test = ["none"] + list(transforms_to_test)

    skip = [t for t in TRANSFORMS if t not in transforms_to_test]
    if verbose:
        print("\n  AUDIT: запускаю эксперимент с преобразованиями:",
              ", ".join(TR_LABEL[t] for t in transforms_to_test))
        if model_key not in REGRESSION_MODELS:
            print(f"  (модель «{MODEL_LABEL[model_key]}» — класса "
                  f"«{MODEL_CLASS[model_key]}»; преобразования не применяются,")
            print("   оценивается только baseline 'none' для отчётности.)")

    run_experiment_cv = _load_chapter2()  # lazy-import тяжёлой машинерии
    df_cv, _, _ = run_experiment_cv(
        X, y, shift=diag.suggested_shift,
        models=[model_key], skip_transforms=skip,
    )

    actual = {}
    for t in transforms_to_test:
        if (model_key, t) in df_cv.index:
            actual[t] = float(df_cv.loc[(model_key, t), "delta_rmse_pct"])

    if verbose:
        print("\n  Прогноз advisor-а vs. фактический результат "
              f"({MODEL_LABEL[model_key]})")
        print("  " + "─" * 70)
        print(f"    {'Преобразование':<14} {'Прогноз':>10} {'Факт':>10} "
              f"{'Δ(факт-прогноз)':>16} {'DM_p':>8}")
        for r in recs:
            actual_d = actual.get(r.transform, float("nan"))
            diff = actual_d - r.predicted_delta_pct
            try:
                dm_p = float(df_cv.loc[(model_key, r.transform), "DM_p"])
                dm_p_str = f"{dm_p:.3f}"
            except (KeyError, ValueError):
                dm_p_str = "—"
            flag = " ✓" if (not np.isnan(actual_d)
                            and np.sign(actual_d) == np.sign(r.predicted_delta_pct)
                            and abs(actual_d) > 1.0) else ""
            print(f"    {TR_LABEL[r.transform]:<14} "
                  f"{r.predicted_delta_pct:>+9.1f}% {actual_d:>+9.1f}%"
                  f" {diff:>+14.1f}pp {dm_p_str:>8}{flag}")
        print("\n    ✓ — направление и масштаб улучшения подтверждены эмпирически")

    return diag, recs, df_cv, actual


# ═════════════════════════════════════════════════════════════════════════════
# 5b. EXTRACT_RULES_FOR_USER — процедурный «вид» KB для пользователя
# ═════════════════════════════════════════════════════════════════════════════
#
# Логика advisor-а на вкладках «Диагностика» и «Рекомендации» — это процедурное
# программирование над KB (full_results_v5.csv): берём γ₁_бин пользователя,
# фильтруем правила KB, ранжируем. НИКАКОЕ обучение моделей не происходит.
#
# extract_rules_for_user() возвращает «срез» KB для конкретного пользователя:
# какие (модель, преобразование) комбинации эмпирически работали на 27
# датасетах из Главы 2 для его бина асимметрии. Это и есть «методические
# рекомендации» в человекочитаемом виде.
# ═════════════════════════════════════════════════════════════════════════════

def extract_rules_for_user(diag: TargetDiagnostics,
                           kb: KnowledgeBase,
                           model_key: Optional[str] = None) -> pd.DataFrame:
    """Возвращает таблицу правил KB, применимых к данным пользователя.

    Правило = одна строка из KB вида (Model, Transform, gamma_bin →
    медиана ΔRMSE, P10/P90, доля значимых улучшений, n_evidence).

    Параметры
    ─────────
    diag       : результат diagnose_target(y) — определяет γ₁_бин
    kb         : KnowledgeBase — источник правил
    model_key  : опционально — отфильтровать только для одной модели

    Возвращает
    ─────────
    DataFrame с колонками (model_label, transform_label, transform_class,
    median_delta_pct, p10_delta_pct, p90_delta_pct, prob_improvement,
    sig_better_rate, n_evidence), отсортированный по медиане ΔRMSE.
    """
    if not hasattr(kb, "_rules") or kb._rules.empty:
        return pd.DataFrame()

    rules = kb._rules.copy()
    rules = rules[rules["gamma_bin"] == diag.gamma_bin].copy()
    if model_key is not None:
        rules = rules[rules["Model_key"] == model_key]

    if rules.empty:
        return rules

    rules["model_label"]     = rules["Model_key"].map(
        lambda k: MODEL_LABEL.get(k, k))
    rules["transform_label"] = rules["Transform_key"].map(
        lambda k: TR_LABEL.get(k, k))
    rules["transform_class"] = rules["Transform_key"].map(
        lambda k: TR_CLASS.get(k, "—"))
    rules["model_class"]     = rules["Model_key"].map(
        lambda k: MODEL_CLASS.get(k, "—"))

    cols = ["model_label", "model_class", "transform_label", "transform_class",
            "median_delta_pct", "p10_delta_pct", "p90_delta_pct",
            "prob_improvement", "sig_better_rate", "n_evidence"]
    cols = [c for c in cols if c in rules.columns]
    rules = rules[cols].sort_values(
        ["model_label", "median_delta_pct"], ascending=[True, True]
    ).reset_index(drop=True)
    return rules


# ═════════════════════════════════════════════════════════════════════════════
# 5c. EMPIRICAL_FULL_AUDIT — обучение ВСЕХ моделей на данных пользователя
# ═════════════════════════════════════════════════════════════════════════════
#
# В отличие от recommend()/recommend_strategy() (KB-lookup, без обучения),
# empirical_full_audit() запускает run_experiment_cv для всех моделей × всех
# применимых преобразований и возвращает РАНЖИРОВАНИЕ по фактическому RMSE
# на hold-out. Это «strategy comparison» в её эмпирической форме.
#
# Используется на вкладке «Сравнение стратегий» в Streamlit. Требует
# chapter2_experiments_v6.py со всеми зависимостями (sklearn, xgboost,
# statsmodels, matplotlib). В облачном режиме — недоступно.
# ═════════════════════════════════════════════════════════════════════════════

def empirical_full_audit(
        X, y,
        models: Optional[List[str]] = None,
        shift: Optional[float] = None,
        progress_callback=None,
        verbose: bool = False,
        ) -> tuple:
    """Обучает ВСЕ модели × применимые преобразования, ранжирует по RMSE.

    Параметры
    ─────────
    X, y               : пользовательский датасет (X — фичи, y — отклик)
    models             : список моделей; default = все из MODELS
    shift              : δ для log/Box-Cox; auto если None
    progress_callback  : функция (i, total, current_model_label) -> None;
                         вызывается перед каждой моделью. Для st.progress в UI.
    verbose            : печать в stdout

    Возвращает
    ──────────
    (diag, df_ranked, winner) где
       diag      : TargetDiagnostics для y
       df_ranked : DataFrame, индексированный (model_key, transform_key),
                   отсортированный по RMSE на hold-out (по возрастанию).
                   Колонки: RMSE, RMSE_train, delta_rmse_pct, DM_p, paired_t_p,
                            cohens_d, model_label, transform_label, model_class,
                            rank.
       winner    : dict с описанием победителя (model + transform + RMSE + ...)
                   или None, если все модели упали.
    """
    diag = diagnose_target(y, shift=shift)
    if models is None:
        models = list(MODELS)

    rec_run = _load_chapter2()  # lazy-load run_experiment_cv

    if verbose:
        print(f"\n  EMPIRICAL AUDIT: обучаю {len(models)} моделей × "
              f"{len(TRANSFORMS)} преобразований на n={len(y):,}, "
              f"p={np.asarray(X).shape[1]}")

    # Запускаем по одной модели за раз — это даёт прогресс в UI
    dfs = []
    for i, m_key in enumerate(models):
        m_label = MODEL_LABEL.get(m_key, m_key)
        if progress_callback is not None:
            progress_callback(i, len(models), m_label)
        if verbose:
            print(f"    [{i+1}/{len(models)}] {m_label}…", end="", flush=True)
        try:
            df_i, _, _ = rec_run(
                X, y, shift=diag.suggested_shift,
                models=[m_key], skip_transforms=(),
            )
            dfs.append(df_i)
            if verbose:
                n_rows = len(df_i.dropna(subset=["RMSE"]))
                print(f" ✓ ({n_rows} результатов)")
        except Exception as e:
            if verbose:
                print(f" ✗ {type(e).__name__}: {str(e)[:60]}")

    if progress_callback is not None:
        progress_callback(len(models), len(models), None)

    if not dfs:
        raise RuntimeError(
            "Empirical audit: все модели завершились с ошибкой. "
            "Проверьте размер датасета, наличие NaN и числовые типы фичей.")

    df_full = pd.concat(dfs)
    df_full = df_full.dropna(subset=["RMSE"]).copy()

    # Ранжирование по RMSE на hold-out (меньше = лучше)
    df_ranked = df_full.sort_values("RMSE", ascending=True).copy()
    df_ranked["rank"] = range(1, len(df_ranked) + 1)
    df_ranked["model_label"] = [
        MODEL_LABEL.get(idx[0], idx[0]) for idx in df_ranked.index]
    df_ranked["transform_label"] = [
        TR_LABEL.get(idx[1], idx[1]) for idx in df_ranked.index]
    df_ranked["model_class"] = [
        MODEL_CLASS.get(idx[0], "—") for idx in df_ranked.index]

    # Победитель
    if df_ranked.empty:
        winner = None
    else:
        top = df_ranked.iloc[0]
        m_key, t_key = df_ranked.index[0]
        dm_p_val = top.get("DM_p", float("nan"))
        winner = {
            "model_key":       m_key,
            "transform_key":   t_key,
            "model_label":     MODEL_LABEL.get(m_key, m_key),
            "transform_label": TR_LABEL.get(t_key, t_key),
            "model_class":     MODEL_CLASS.get(m_key, "—"),
            "RMSE":            float(top["RMSE"]),
            "delta_rmse_pct":  float(top.get("delta_rmse_pct", float("nan"))),
            "DM_p":            float(dm_p_val) if not np.isnan(dm_p_val) else None,
            "rank":            1,
        }

    if verbose and winner is not None:
        print(f"\n  🏆 Победитель: {winner['model_label']} + "
              f"{winner['transform_label']}  →  RMSE = {winner['RMSE']:.4g}")

    return diag, df_ranked, winner


# ═════════════════════════════════════════════════════════════════════════════
# 6. PRETTY OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def print_report(diag: TargetDiagnostics, recs: List[Recommendation],
                 model_key: str) -> None:
    """Печатает диагностику + рекомендации в формате терминального отчёта."""
    print("\n" + "═" * 72)
    print(f"  ОТЧЁТ ADVISOR-А для модели «{MODEL_LABEL[model_key]}»  "
          f"(класс: {MODEL_CLASS[model_key]})")
    print("═" * 72)

    print("\n  Диагностика отклика Y (без обучения модели)")
    print("  " + "─" * 60)
    print(f"    n = {diag.n:,}")
    print(f"    γ₁  = {diag.gamma1:+.3f}    ← {diag.gamma_bin}")
    print(f"    γ₂  = {diag.excess_kurt:+.3f} (excess kurtosis; 0 для N(·))")
    print(f"    E[Y]={diag.mean:.4g}  Median={diag.median:.4g}  "
          f"Sd={diag.std:.4g}")
    print(f"    Min ={diag.minv:.4g}  Max ={diag.maxv:.4g}")
    if diag.zeros_pct > 0:
        print(f"    Нулей в Y: {diag.zeros_pct:.1f}%")
    if diag.neg_pct > 0:
        print(f"    Отрицательных в Y: {diag.neg_pct:.1f}%")
    if diag.suggested_shift > 0:
        print(f"    Реком. сдвиг δ = {diag.suggested_shift:.3g} "
              f"(для log/Box-Cox)")
    sw_verdict = ("нормальность отвергается" if diag.shapiro_p < ALPHA
                  else "нормальность не отвергается")
    print(f"    Shapiro–Wilk p = {diag.shapiro_p:.4g}  ({sw_verdict})")
    if diag.boxcox_lambda is not None:
        print(f"    Box-Cox λ̂ = {diag.boxcox_lambda:.3f}  "
              f"95% ДИ: [{diag.boxcox_ci_lo:.3f}; {diag.boxcox_ci_hi:.3f}]")
        if diag.lrt_log_p is not None:
            v = "ОТВЕРГ" if diag.lrt_log_p < ALPHA else "не отверг."
            print(f"    LRT H₀: λ=0 (log)         χ²={diag.lrt_log_chi2:6.2f}  "
                  f"p={diag.lrt_log_p:.4g}  → {v}")
        if diag.lrt_none_p is not None:
            v = "ОТВЕРГ" if diag.lrt_none_p < ALPHA else "не отверг."
            print(f"    LRT H₀: λ=1 (без преобр.) χ²={diag.lrt_none_chi2:6.2f}  "
                  f"p={diag.lrt_none_p:.4g}  → {v}")
    else:
        print("    Box-Cox: недоступно (Y+δ имеет неположительные значения)")

    print(f"\n  Рекомендации (Top-{len(recs)}) по медианному ΔRMSE_test%")
    print("  " + "─" * 60)
    for r in recs:
        flag = "✓ ПРИМ." if r.applicable else "✗ НЕ ПРИМ."
        print(f"\n    [#{r.rank}] {flag}  {r.transform_label}  "
              f"— {r.transform_class}")
        # ── БЛОК 5 ТЗ: показываем диапазон [P10, P90], а не точечный прогноз ──
        if (not np.isnan(r.delta_p10)) and (not np.isnan(r.delta_p90)):
            print(f"        прогноз ΔRMSE: медиана = {r.delta_p50:+5.1f}%,  "
                  f"диапазон [P10, P90] = [{r.delta_p10:+5.1f}%, "
                  f"{r.delta_p90:+5.1f}%]")
            print(f"        P(улучшение) = {r.prob_improvement*100:3.0f}%   "
                  f"P(значимое улучшение) = {r.prob_significant*100:3.0f}%   "
                  f"n_evidence = {r.n_evidence}")
            # Маркер «надёжности» по ширине интервала
            ci_width = r.delta_p90 - r.delta_p10
            if ci_width < 10.0:
                ci_label = "узкий (надёжная рекомендация)"
            elif ci_width < 25.0:
                ci_label = "умеренный"
            else:
                ci_label = "широкий (рискованная — большой разброс по бину)"
            print(f"        Ширина интервала [P10, P90] = "
                  f"{ci_width:.1f}pp  ({ci_label})")
        else:
            print(f"        прогноз ΔRMSE = {r.predicted_delta_pct:+6.1f}%   "
                  f"P(сущ. лучше) = {r.sig_better_rate*100:3.0f}%   "
                  f"n_evidence = {r.n_evidence}")
        print(f"        {r.rationale}")
        for w in r.warnings:
            print(f"        ⚠ {w}")
    print()


def save_markdown_report(diag: TargetDiagnostics, recs: List[Recommendation],
                          model_key: str, path: Path) -> None:
    """Сохраняет отчёт в Markdown для вставки в ВКР."""
    lines: List[str] = []
    lines.append(f"# Отчёт advisor-а — модель «{MODEL_LABEL[model_key]}»\n")
    lines.append(f"_Класс модели: {MODEL_CLASS[model_key]}_  \n")
    lines.append(f"_Бин асимметрии: {diag.gamma_bin}_\n")

    lines.append("## 1. Диагностика отклика Y\n")
    lines.append(f"| Статистика | Значение |\n|---|---|")
    lines.append(f"| n | {diag.n:,} |")
    lines.append(f"| γ₁ (skewness) | {diag.gamma1:+.3f} |")
    lines.append(f"| γ₂ (excess kurtosis) | {diag.excess_kurt:+.3f} |")
    lines.append(f"| E[Y] | {diag.mean:.4g} |")
    lines.append(f"| Median | {diag.median:.4g} |")
    lines.append(f"| Sd | {diag.std:.4g} |")
    lines.append(f"| Min / Max | {diag.minv:.4g} / {diag.maxv:.4g} |")
    lines.append(f"| Нулей % | {diag.zeros_pct:.2f} |")
    lines.append(f"| Отрицательных % | {diag.neg_pct:.2f} |")
    lines.append(f"| Shapiro–Wilk p | {diag.shapiro_p:.4g} |")
    if diag.boxcox_lambda is not None:
        lines.append(f"| Box-Cox λ̂ | {diag.boxcox_lambda:.3f} "
                     f"[{diag.boxcox_ci_lo:.3f}; {diag.boxcox_ci_hi:.3f}] |")
        if diag.lrt_log_p is not None:
            lines.append(f"| LRT H₀: λ=0 (log), p | {diag.lrt_log_p:.4g} |")
        if diag.lrt_none_p is not None:
            lines.append(f"| LRT H₀: λ=1, p | {diag.lrt_none_p:.4g} |")

    lines.append("\n## 2. Рекомендации\n")
    lines.append("| # | Применимо | Преобр. | Класс | Медиана ΔRMSE% | "
                 "[P10; P90] | P(улучш.) | P(знач.) | n_evidence |\n"
                 "|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        ap = "✓" if r.applicable else "✗"
        if (not np.isnan(r.delta_p10)) and (not np.isnan(r.delta_p90)):
            range_str = f"[{r.delta_p10:+.1f}; {r.delta_p90:+.1f}]"
            prob_imp_str = f"{r.prob_improvement*100:.0f}%"
        else:
            range_str = "—"
            prob_imp_str = "—"
        lines.append(f"| {r.rank} | {ap} | {r.transform_label} | "
                     f"{r.transform_class} | "
                     f"{r.predicted_delta_pct:+.1f}% | "
                     f"{range_str} | {prob_imp_str} | "
                     f"{r.sig_better_rate*100:.0f}% | {r.n_evidence} |")

    lines.append("\n## 3. Обоснование и предупреждения\n")
    for r in recs:
        lines.append(f"### [{r.rank}] {r.transform_label}\n")
        lines.append(r.rationale)
        if r.warnings:
            lines.append("\n**Предупреждения:**")
            for w in r.warnings:
                lines.append(f"- {w}")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# 7. ДЕМОНСТРАЦИЯ (main)
# ═════════════════════════════════════════════════════════════════════════════

def _load_demo_data():
    """Diamonds через pydataset, либо fallback на лог-нормальную синтетику."""
    try:
        from pydataset import data as pdata
        d = pdata("diamonds").dropna()
        d = d[(d["x"] > 0) & (d["y"] > 0) & (d["z"] > 0)]
        d = d[(d["y"] < 20) & (d["z"] < 20)]
        cut_o = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        col_o = ["J", "I", "H", "G", "F", "E", "D"]
        cla_o = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
        d["cut_e"]   = d["cut"].map({v: i for i, v in enumerate(cut_o)})
        d["col_e"]   = d["color"].map({v: i for i, v in enumerate(col_o)})
        d["cla_e"]   = d["clarity"].map({v: i for i, v in enumerate(cla_o)})
        feats = ["carat", "cut_e", "col_e", "cla_e", "depth", "table",
                 "x", "y", "z"]
        X = d[feats].values.astype(float)
        y = d["price"].values.astype(float)
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y), size=min(3000, len(y)), replace=False)
        return X[idx], y[idx], "Diamonds (pydataset, n=3000)"
    except Exception:
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.standard_normal((2000, 5))
        log_y = 2.0 + X @ np.array([0.3, 0.2, -0.1, 0.05, 0.0]) + \
                0.5 * rng.standard_normal(2000)
        return X, np.exp(log_y), "synthetic log-normal (n=2000, σ=0.5)"


def demo(kb_path: Optional[str] = None, run_audit: bool = True):
    print("\n" + "█" * 72)
    print("  ВКР — ГЛАВА 3. TEST STAND / ADVISOR")
    print("  Pre-training рекомендации нелинейных преобразований Y")
    print("█" * 72)

    # ── 1. Загрузка KB ──
    print("\n  [1/4] Загрузка KnowledgeBase")
    kb_candidates = []
    if kb_path: kb_candidates.append(Path(kb_path))
    kb_candidates += [
        RESULTS_DIR / "full_results_v5.csv",
        RESULTS_DIR / "full_results_v4.csv",
        Path("results") / "full_results_v5.csv",
        Path("results") / "full_results_v4.csv",
    ]
    kb = None
    for p in kb_candidates:
        if p.exists():
            try:
                kb = KnowledgeBase.from_csv(p)
                print(f"  ✓ KB загружен из {p}")
                break
            except Exception as e:
                print(f"  [warn] {p}: {type(e).__name__}: {e}")
    if kb is None:
        kb = KnowledgeBase.from_defaults()
        print(f"  ⚠ Результаты Главы 2 не найдены — использован "
              f"априорный (literature-informed) KB.")
        print(f"    Для production-режима сначала запустите "
              f"chapter2_experiments_v5.py.")
    kb.summary()

    # Сохраняем KB как CSV
    kb_out = ADVISOR_DIR / "knowledge_base.csv"
    kb.to_csv(kb_out)
    print(f"  → {kb_out}")

    # ── 2. Загрузка демонстрационного датасета ──
    print("\n  [2/4] Демо-датасет")
    X, y, data_label = _load_demo_data()
    print(f"  Датасет: {data_label}    γ₁={float(skew(y)):+.3f}")

    # ── 3. Рекомендации для 3 моделей ──
    print(f"\n  [3/4] Рекомендации для 3 моделей разного класса")
    for mkey in ("linear", "rf", "mlp"):
        diag, recs = recommend(y, mkey, kb=kb, top_k=3, verbose=True)
        # сохраняем markdown-отчёт
        md_path = ADVISOR_DIR / f"report_{mkey}_demo.md"
        save_markdown_report(diag, recs, mkey, md_path)
        print(f"  → MD-отчёт: {md_path}")

    # ── 4. Empirical audit (только для linear, чтобы не ждать слишком долго) ──
    if run_audit:
        print(f"\n  [4/4] Эмпирический AUDIT (linear; 5-fold CV)")
        diag, recs, df_cv, actual = audit(X, y, model_key="linear", kb=kb)
        audit_csv = ADVISOR_DIR / "audit_linear_demo.csv"
        df_cv.to_csv(audit_csv, encoding="utf-8-sig")
        print(f"  → {audit_csv}")

    print("\n" + "─" * 72)
    print(f"  Все артефакты: ./{ADVISOR_DIR}/")
    print("─" * 72 + "\n")


if __name__ == "__main__":
    demo()
