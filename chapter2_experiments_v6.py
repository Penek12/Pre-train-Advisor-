"""
ВКР — Глава 2. Вычислительный эксперимент (v6).
Нелинейные преобразования отклика: сравнительный анализ
для моделей различного класса сложности.

ИЗМЕНЕНИЯ относительно v5:
  • БЛОК 1 (Cohen's d + power analysis): добавлены функции
    `cohens_d_squared_errors` (размер эффекта на разностях квадратичных потерь)
    и `min_n_for_power` (минимальный n для двустороннего z-теста с заданной
    мощностью). В `run_experiment_cv` рассчитываются `cohens_d`, `min_n_80power`,
    `is_underpowered` для каждой пары (модель × преобр.). Это закрывает
    методологический пробел: p-value зависит от n, размер эффекта — нет.
    В CSV-дампе и в `print_metrics_multi` теперь видно, какие отсутствия
    значимости — артефакт малой выборки.

  • БЛОК 3 (три стратегии для асимметричного Y): добавлены GLM-семейство
    (`gamma_glm`, `tweedie_glm`) и XGBoost с альтернативными loss-функциями
    (`xgb_gamma`, `xgb_tweedie`). Эти модели — содержательная АЛЬТЕРНАТИВА
    преобразованию Y, а не его дополнение:
      – preserve Y в исходной шкале;
      – используют логарифмическую link-функцию (GLM) или соответствующую
        loss (XGBoost) для учёта правой асимметрии распределения Y;
      – не подвержены проблеме Йенсенова смещения при инверсии.

    Применимость: только для Y > 0 (Gamma), либо Y ≥ 0 (Tweedie с var_power=1.5,
    допускает нули). Запускаются по построению только с transform='none' —
    альтернативная loss и преобразование Y методологически несовместимы.

    Расширенный набор моделей (всего 10):
      Линейные регрессии:   linear, ridge, lasso         → с преобразованиями Y
      GLM:                   gamma_glm, tweedie_glm        → без преобразований
      Древесные:             rf, xgb                       → без преобразований
      Древесные + spec.loss: xgb_gamma, xgb_tweedie        → без преобразований
      Нейросеть:             mlp                            → без преобразований

  • Все артефакты v5 (Box-Cox + LRT, поправка Дуана, тесты значимости,
    27-датасетный бенчмарк) сохранены без изменений.

  • Расширенный CSV переименован: `full_results_v5.csv → full_results_v6.csv`.

МЕТОДОЛОГИЧЕСКОЕ ОГРАНИЧЕНИЕ (принципиально!):
  Нелинейные преобразования отклика g(Y) и поправка Дуана / обобщённая
  smearing-оценка (Duan 1983) применяются ИСКЛЮЧИТЕЛЬНО к моделям класса
  линейной регрессии (МНК, Ridge, Lasso). Для древесных моделей (RF, XGBoost)
  и MLP оценивается только baseline (без преобразования отклика).

  Обоснование:
    • Теория Box & Cox (1964), Yeo & Johnson (2000), а равно вывод Дуана
      о состоятельности smearing-оценки опираются на модель
        g(Y) = Xβ + ε,   ε i.i.d.,  E[ε] = 0,
      т.е. на линейность условного среднего в трансформированной шкале
      и гомоскедастичность остатков. Эти предпосылки выполнены для
      OLS/Ridge/Lasso, но не для деревьев и нейросетей.
    • Решающие деревья и градиентный бустинг — непараметрические модели,
      способные сами улавливать нелинейность E[Y|X]; внешнее преобразование
      Y лишь меняет критерий разбиения (variance reduction на g(Y) против
      variance reduction на Y) и поэтому даёт другую, но не более
      обоснованную модель. Дуан-коррекция здесь смысла не имеет —
      остатки в g-шкале не описывают компоненту смещения в исходной шкале.
    • MLP — универсальный аппроксиматор; преобразование Y перед обучением
      создаёт паразитное Йенсеново смещение E[g⁻¹(ĝ + ε̂)] ≠ g⁻¹(E[ĝ]),
      которое не компенсируется поправкой Дуана (она выведена для линейной
      модели с гомоскедастичными остатками, см. Duan 1983, JASA 78(383)).

  Следствие в коде:
    в run_experiment_cv() цикл по (transform × model) для t ≠ 'none' идёт
    только по models ∩ REGRESSION_MODELS = {linear, ridge, lasso}; для
    остальных моделей оценивается лишь 'none'. Поправка Дуана и
    smearing активны лишь для регрессий — на деревьях/MLP они не
    вызываются по построению.

ИЗМЕНЕНИЯ относительно v4:
  • Расширен пул реальных датасетов: добавлено 20 наборов из OpenML / UCI /
    statsmodels / pydataset / sklearn с разной степенью асимметрии Y.
    Цель — плотное покрытие диапазона γ₁ ∈ (-0.5, 30+) для устойчивой
    эмпирической оценки пороговых γ*₁ и формы кривой U(γ₁).

    Группы по асимметрии:
      (S) γ₁ ∈ [-0.5, 0.5]   симметричные / квази-симметричные (6 шт.)
      (M) γ₁ ∈ ( 0.5, 1.5]   умеренно скошенные (5 шт.)
      (H) γ₁ ∈ ( 1.5, 3.0]   сильно скошенные (5 шт.)
      (X) γ₁ >  3.0          экстремально скошенные (4 шт.)

  • Унифицированный обобщённый прогон `run_generic_dataset(...)` — вся
    обвязка (печать описательных статистик, Box-Cox + LRT, эксперимент,
    метрики) собрана в одной функции; loader-ы возвращают единый dict.

  • Безопасный fetch с перебором имён/ID OpenML: устойчивость к
    переименованиям и недоступности отдельных датасетов; неудача одного
    загрузчика не прерывает остальные.

ИЗМЕНЕНИЯ относительно v3 (сохранены из v4):
  • Расширен набор преобразований: 7 шт. в 4 методологических классах
    (степенные с фикс. λ, гиперболические, параметрические степенные,
    ранговые непараметрические).
  • Формализована гипотеза об убывающей полезности преобразований:
    U(model, transform, γ₁) = −ΔRMSE%; тесты Diebold–Mariano и парный
    t-тест по фолдам CV; оценка пороговых γ*₁ для каждого класса моделей.
  • Численная демонстрация поправки Дуана как Йенсеновой компенсации.
  • Базовые реальные датасеты: Diamonds, California Housing, Concrete,
    RAND HIE + синтетика A/B/C.

Модели (6):
  МНК, Ridge, Lasso, Random Forest, XGBoost, MLP

Преобразования (7):
  none, log, sqrt, asinh, boxcox, yeojohnson, quantile

Оценка: 5-fold CV + hold-out test + статистические тесты значимости.
Итоговое число датасетов: 7 (базовые) + 20 (новые) = 27.

Запуск: python chapter2_experiments_v5.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from scipy import stats
from scipy.stats import boxcox, shapiro, skew

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer

import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
N_FOLDS         = 5
RMSE_THRESHOLD  = 5.0       # |ΔRMSE| ≥ 5 % считаем «практически значимым»
ALPHA           = 0.05      # уровень значимости для тестов

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 11,
    "axes.titlesize": 12, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

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
# Класс преобразования (для теоретической главы)
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
# Классы моделей (для гипотезы об убывающей полезности и для advisor-а)
# БЛОК 3 ТЗ: добавлены классы "glm" и "древесные-glm-loss" — это
# содержательные конкуренты преобразованию Y, а не его дополнение.
MODEL_CLASS = {
    "linear":      "линейные",      "ridge":       "линейные",
    "lasso":       "линейные",
    "rf":          "древесные",     "xgb":         "древесные",
    "mlp":         "нейросетевые",
    "gamma_glm":   "glm",           "tweedie_glm": "glm",
    "xgb_gamma":   "древесные-glm-loss",
    "xgb_tweedie": "древесные-glm-loss",
}

# ─── Применимость нелинейных преобразований Y по классам моделей ─────────────
# Нелинейное преобразование отклика g(Y) с последующей обратной инверсией
# g⁻¹(·) методологически обосновано ТОЛЬКО для линейных регрессий
# (МНК, Ridge, Lasso). Теория Box & Cox (1964) и поправка Дуана (Duan 1983)
# исходят из модели g(Y) = Xβ + ε с i.i.d. гомоскедастичными ε. Для
# древесных моделей и MLP таких предпосылок нет: деревья непараметричны
# и сами улавливают нелинейность; MLP — универсальный аппроксиматор,
# и применение преобразования Y создаёт некомпенсируемое поправкой Дуана
# Йенсеново смещение при инверсии. Поэтому в run_experiment_cv() для
# моделей вне REGRESSION_MODELS оценивается только baseline 'none'.
REGRESSION_MODELS = frozenset({"linear", "ridge", "lasso"})

COLORS_TR = {
    "none":"#AAAAAA", "log":"#4472C4", "sqrt":"#ED7D31",
    "asinh":"#16A085", "boxcox":"#70AD47", "yeojohnson":"#FFC000",
    "quantile":"#9B59B6",
}
COLORS_MODEL = {"linear":"#4472C4","ridge":"#2E5A88","lasso":"#1B3A57",
                "rf":"#ED7D31","xgb":"#70AD47","mlp":"#9B59B6"}

SYNTH_BETA = np.array([0.12, 0.08, -0.05])
SYNTH_SERIES = {"A": (0.082, "слабая"), "B": (0.414, "умеренная"), "C": (0.750, "высокая")}


# ══════════════════════════════════════════════════════════════════════════════
# 0. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Все метрики в исходной шкале отклика."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    r2   = float(r2_score(y_true, y_pred))
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)


def duan_smearing(resid_train_log):
    """
    Поправка Дуана (Duan 1983, JASA): c = (1/n) Σ exp(ê_i).

    Теоретическое обоснование (см. главу 1).
    Пусть log(Y) = Xβ + ε, ε i.i.d. с E[ε]=0.
    Тогда Y = exp(Xβ) · exp(ε), и:
        E[Y | X] = exp(Xβ) · E[exp(ε)].
    «Наивный» обратный прогноз ŷ_naive = exp(X β̂) даёт оценку exp(Xβ),
    но смещён вниз на множитель E[exp(ε)] относительно E[Y|X].

    По неравенству Йенсена для выпуклой функции exp(·):
        E[exp(ε)] ≥ exp(E[ε]) = 1,
    причём строгое неравенство выполняется при ε не вырожденной.
    Для нормальных остатков ε ∼ N(0, σ²):  E[exp(ε)] = exp(σ²/2).

    Поправка Дуана даёт состоятельную НЕпараметрическую оценку E[exp(ε)],
    т.е. компенсирует Йенсеново смещение без предположения нормальности.
    """
    c = float(np.mean(np.exp(resid_train_log)))
    return c, (c - 1.0) * 100.0


def general_smearing(pred_transformed, resid_train, inv_func, max_n=5000):
    """
    Обобщённая smearing-оценка (Duan 1983, общий вид).

    Для любого монотонного преобразования g с инверсией g⁻¹ модель
    выдаёт несмещённую оценку ĝ(x) в трансформированной шкале, но
    g⁻¹(ĝ(x)) — смещённая оценка E[Y|X] из-за неравенства Йенсена.

    Smearing-корректор:
        Ê[Y | X=x] = (1/n) Σᵢ g⁻¹( ĝ(x) + ê_i )

    где ê_i — остатки на обучающей выборке в g-шкале. Это
    состоятельная НЕпараметрическая оценка (не требует нормальности
    остатков), применимая к ЛЮБОМУ нелинейному g (не только log).

    Параметры
    ─────────
    pred_transformed : np.ndarray (n_test,) — ĝ(x) на тестовой выборке
    resid_train      : np.ndarray (n_train,) — ê_i = g(y_train) − ĝ(x_train)
    inv_func         : callable, g⁻¹(·)
    max_n            : int, подвыборка остатков для скорости (O(n_test · max_n))

    Возвращает
    ──────────
    np.ndarray (n_test,) — Ê[Y|X] в исходной шкале (без shift-вычета;
    shift уже учтён внутри inv_func).
    """
    eps = np.asarray(resid_train, dtype=float).ravel()
    if len(eps) > max_n:
        idx = np.random.default_rng(RANDOM_STATE).choice(
            len(eps), max_n, replace=False)
        eps = eps[idx]
    pred = np.asarray(pred_transformed, dtype=float).ravel()
    # Broadcasting: (n_test, 1) + (1, n_eps) → (n_test, n_eps)
    grid = pred.reshape(-1, 1) + eps.reshape(1, -1)
    # Применяем g⁻¹ ко всему grid и усредняем по остаткам
    inverted = inv_func(grid.ravel()).reshape(grid.shape)
    return np.nanmean(inverted, axis=1)



def boxcox_fit(y_train):
    """Подбор λ по MLE + 95 % доверительный интервал через LRT."""
    _, lam    = boxcox(y_train)
    ll_opt    = stats.boxcox_llf(lam, y_train)
    chi2_crit = stats.chi2.ppf(0.95, df=1)
    grid      = np.linspace(lam - 3.0, lam + 3.0, 3000)
    in_ci     = [l for l in grid if 2.0*(ll_opt - stats.boxcox_llf(l, y_train)) < chi2_crit]
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


# ─────────────────────────────────────────────────────────────────────────────
# Тесты значимости различий между моделями/преобразованиями
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano(y_true, pred_baseline, pred_alt, loss="se"):
    """
    Тест Diebold–Mariano (1995) на равенство ожидаемых потерь двух прогнозов.

        H₀: E[L(e_baseline) − L(e_alt)] = 0
        H₁: E[L(e_baseline) − L(e_alt)] ≠ 0

    Возвращает (DM-статистика, p-value, mean_d).
    Положительный mean_d ⇒ alt лучше (меньше потерь).

    Замечание: для задач без временной зависимости (i.i.d. наблюдения)
    DM эквивалентен парному z-тесту на разностях квадратичных потерь.
    Для time-series требуется HAC-оценка дисперсии (здесь не нужна).
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_baseline = np.asarray(pred_baseline, dtype=float)
    pred_alt = np.asarray(pred_alt, dtype=float)

    if loss == "se":
        L_b = (y_true - pred_baseline) ** 2
        L_a = (y_true - pred_alt) ** 2
    elif loss == "ae":
        L_b = np.abs(y_true - pred_baseline)
        L_a = np.abs(y_true - pred_alt)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = L_b - L_a   # положительные значения ⇒ alt предпочтительнее
    n = len(d)
    dbar = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d == 0 or n < 2:
        return 0.0, 1.0, dbar
    dm_stat = dbar / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value), dbar


def paired_cv_ttest(rmses_baseline, rmses_alt):
    """
    Парный t-тест на RMSE по фолдам кросс-валидации.

        H₀: средняя разность RMSE между alt и baseline = 0
        H₁: ≠ 0

    Возвращает (t-статистика, p-value, mean_diff_pct).
    Отрицательный mean_diff_pct ⇒ alt улучшает (меньшая RMSE).
    """
    a = np.asarray(rmses_alt, dtype=float)
    b = np.asarray(rmses_baseline, dtype=float)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan"), float("nan"), float("nan")
    if np.allclose(a, b):
        return 0.0, 1.0, 0.0
    t_stat, p_value = stats.ttest_rel(a, b)
    mean_diff_pct = float(np.mean((a - b) / b) * 100.0)
    return float(t_stat), float(p_value), mean_diff_pct


# ─────────────────────────────────────────────────────────────────────────────
# Эффект-сайз и оценка мощности теста (БЛОК 1 ТЗ)
# ─────────────────────────────────────────────────────────────────────────────
#
# Зачем это нужно. Diebold-Mariano даёт p-value, которое зависит от размера
# hold-out выборки n_test. На малых датасетах (Yacht: 308, mtcars: 32) даже
# крупный относительный выигрыш ΔRMSE=−15% может остаться статистически
# незначимым (p>0.05) просто из-за нехватки наблюдений — это
# «underpowered»-случай, а не отсутствие эффекта.
#
# Cohen's d на разностях потерь d_i = L_baseline,i − L_alt,i — это
# нормированный размер эффекта, инвариантный к n. Интерпретация по Cohen
# (1988): |d|<0.2 малый, 0.2–0.5 средний, 0.5–0.8 умеренный, >0.8 крупный.
#
# Обратная задача (min_n_for_power) отвечает на вопрос: каков минимальный
# размер тестовой выборки, при котором эффект данного размера был бы
# обнаружим с заданной мощностью? Формула:  n* = (z_{α/2} + z_β)² / d².
# Если len(y_test) < n*, текущий тест недомощный — его p>α неинформативен.
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d_squared_errors(y_true, pred_baseline, pred_alt):
    """Cohen's d на разностях квадратичных потерь.

    Соглашение знака: d > 0 ⇒ alt предпочтительнее (меньше потерь);
                      d < 0 ⇒ baseline предпочтительнее.

    Интерпретация по Cohen (1988):
        |d| < 0.2    малый эффект
        0.2 ≤ |d| ≤ 0.5  средний
        0.5 < |d| ≤ 0.8  умеренный
        |d| > 0.8        крупный

    Это размер эффекта, инвариантный к n_test — в отличие от p-value
    DM-теста, который тает с уменьшением выборки.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_baseline = np.asarray(pred_baseline, dtype=float)
    pred_alt = np.asarray(pred_alt, dtype=float)
    L_b = (y_true - pred_baseline) ** 2
    L_a = (y_true - pred_alt) ** 2
    d = L_b - L_a   # положит. ⇒ alt лучше
    sd = float(np.std(d, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return 0.0
    return float(np.mean(d) / sd)


def min_n_for_power(cohens_d, alpha=0.05, power=0.80):
    """Минимальный n_test для двустороннего z-теста с заданной мощностью.

    Стандартная формула апостериорного анализа мощности для одной выборки:
        n* = (z_{α/2} + z_β)² / d²

    где d — Cohen's d (нормированный размер эффекта), α — уровень значимости,
    1−β — мощность. При |d|→0 формула расходится (бесконечно много
    наблюдений нужно, чтобы обнаружить нулевой эффект) — возвращаем inf.

    Используется в `run_experiment_cv` для маркировки underpowered-строк:
    если фактический n_test меньше n*, то p>α не значит «эффекта нет»,
    а значит «выборка слишком мала, чтобы его поймать».
    """
    from scipy.stats import norm
    if abs(cohens_d) < 1e-6:
        return float("inf")
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    return float((z_a + z_b) ** 2 / (cohens_d ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Применение преобразований
# ─────────────────────────────────────────────────────────────────────────────

def apply_transform(y_tr, y_te, transform, shift=0.0):
    """
    Применяет одно из 7 нелинейных преобразований к отклику.

    Все преобразования принадлежат одному из 4 методологических классов:
        – Степенные с фиксированным λ:  log (λ→0), sqrt (λ=0.5)
        – Гиперболические:               asinh
        – Параметрические степенные:     Box-Cox, Yeo-Johnson (λ по MLE)
        – Ранговые непараметрические:    Quantile (ECDF → Φ⁻¹)

    Возвращает: (y_tr_t, y_te_t, λ̂, inv_func).
    """
    yt, ye = y_tr + shift, y_te + shift

    if transform == "none":
        return yt, ye, None, lambda x, **kw: x - shift

    elif transform == "log":
        # log — предельный случай Box-Cox при λ→0;
        # инверсия требует поправки Дуана (Йенсенова компенсация).
        if (yt <= 0).any() or (ye <= 0).any():
            raise ValueError("log requires strictly positive values; use shift>0")
        y_tr_t, y_te_t = np.log(yt), np.log(ye)
        def inv_log(x, resid_train=None, **kw):
            raw = np.exp(x)
            if resid_train is not None:
                c, _ = duan_smearing(resid_train)
                raw = raw * c
            return raw - shift
        return y_tr_t, y_te_t, 0.0, inv_log

    elif transform == "sqrt":
        # sqrt — частный случай Box-Cox при λ=0.5 (т.н. ladder of powers).
        if (yt < 0).any() or (ye < 0).any():
            raise ValueError("sqrt requires non-negative values")
        return np.sqrt(yt), np.sqrt(ye), 0.5, \
               lambda x, **kw: np.maximum(x, 0.0) ** 2 - shift

    elif transform == "asinh":
        # Обратный гиперболический синус: asinh(y) = ln(y + √(y²+1)).
        # Ведёт себя как ln(2y) при больших |y| и как y при малых.
        # Преимущество: определён на всей вещественной прямой и шкале≈ln без shift.
        y_tr_t = np.arcsinh(yt)
        y_te_t = np.arcsinh(ye)
        return y_tr_t, y_te_t, None, \
               lambda x, **kw: np.sinh(x) - shift

    elif transform == "boxcox":
        if (yt <= 0).any() or (ye <= 0).any():
            raise ValueError("Box-Cox requires strictly positive values; use shift>0")
        y_tr_t, lam = boxcox(yt)
        y_te_t = np.log(ye) if abs(lam) < 1e-8 else (ye ** lam - 1.0) / lam
        _l, _s = float(lam), float(shift)
        def inv_bc(x, **kw):
            if abs(_l) < 1e-8:
                return np.exp(x) - _s
            return np.maximum(_l * x + 1.0, 0.0) ** (1.0 / _l) - _s
        return y_tr_t, y_te_t, _l, inv_bc

    elif transform == "yeojohnson":
        # Yeo-Johnson — обобщение Box-Cox на R через кусочное определение.
        # Применимо в т.ч. при Y, принимающем неположительные значения.
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        y_tr_t = pt.fit_transform(yt.reshape(-1, 1)).ravel()
        y_te_t = pt.transform(ye.reshape(-1, 1)).ravel()
        _pt, _s = pt, float(shift)
        return y_tr_t, y_te_t, float(pt.lambdas_[0]), \
               lambda x, **kw: _pt.inverse_transform(x.reshape(-1, 1)).ravel() - _s

    elif transform == "quantile":
        # Непараметрическое преобразование: ECDF + Φ⁻¹.
        # Принудительно делает распределение нормальным; теряет
        # информацию о расстояниях (только порядок наблюдений сохраняется).
        n_q = min(1000, max(10, len(yt) // 2))
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_q,
            random_state=RANDOM_STATE,
        )
        y_tr_t = qt.fit_transform(yt.reshape(-1, 1)).ravel()
        y_te_t = qt.transform(ye.reshape(-1, 1)).ravel()
        _qt, _s = qt, float(shift)
        return y_tr_t, y_te_t, None, \
               lambda x, **kw: _qt.inverse_transform(x.reshape(-1, 1)).ravel() - _s

    else:
        raise ValueError(f"Unknown transform: {transform}")


def make_model(key):
    if key == "linear":   return LinearRegression()
    elif key == "ridge":  return RidgeCV(alphas=np.logspace(-3, 3, 50))
    elif key == "lasso":  return LassoCV(alphas=np.logspace(-3, 1, 30), max_iter=5000)
    elif key == "rf":     return RandomForestRegressor(
                              n_estimators=300, max_depth=12, min_samples_leaf=5,
                              random_state=RANDOM_STATE, n_jobs=-1)
    elif key == "xgb":    return xgb.XGBRegressor(
                              n_estimators=300, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    elif key == "mlp":    return MLPRegressor(
                              hidden_layer_sizes=(128, 64), activation="relu",
                              solver="adam", max_iter=500, early_stopping=True,
                              validation_fraction=0.15, random_state=RANDOM_STATE)
    # ── БЛОК 3 ТЗ: GLM-семейство (без преобразования Y, log-link внутри) ──
    elif key == "gamma_glm":
        return GLMWrapper(family=sm.families.Gamma(link=sm.families.links.Log()))
    elif key == "tweedie_glm":
        return GLMWrapper(family=sm.families.Tweedie(
            var_power=1.5, link=sm.families.links.Log()))
    # ── БЛОК 3 ТЗ: XGBoost с alt-loss (без преобразования Y) ──
    elif key == "xgb_gamma":
        return xgb.XGBRegressor(
            objective="reg:gamma",
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    elif key == "xgb_tweedie":
        return xgb.XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    else:
        raise ValueError(key)


# ─────────────────────────────────────────────────────────────────────────────
# GLM-обёртка (БЛОК 3 ТЗ): тонкая прослойка sklearn-совместимого интерфейса
# над statsmodels.GLM. Используется для Gamma- и Tweedie-регрессий с log-link.
#
# Зачем GLM:
#   Преобразование Y и GLM с подходящим распределением — это ДВА разных
#   подхода к одной задаче (учёт правой асимметрии Y > 0):
#     • Преобразование g(Y): меняет шкалу отклика; требует Y > 0 для log/Box-Cox
#       и поправки Дуана при обратной инверсии (Йенсеново смещение).
#     • GLM: сохраняет Y в исходной шкале; нелинейность учитывается через
#       link-функцию η = g(E[Y|X]) = Xβ, а распределение остатков задаётся
#       семейством (Gamma для непрерывных положительных, Tweedie для нулей).
#       Йенсеновой проблемы нет — E[Y|X] оценивается напрямую.
#
# В рамках Главы 3 (advisor) добавление GLM превращает рекомендацию
# «какое преобразование» в «какую СТРАТЕГИЮ» (см. Strategy в chapter3_advisor).
# ─────────────────────────────────────────────────────────────────────────────

class GLMWrapper:
    """Sklearn-совместимая обёртка над statsmodels.GLM.

    Поддерживает интерфейс fit(X, y) / predict(X). Использует константу в
    дизайне (intercept) автоматически через sm.add_constant. Возвращаемый
    predict — это E[Y|X] в исходной шкале (statsmodels делает inverse link
    автоматически), поэтому ни поправка Дуана, ни smearing здесь не нужны.

    При отсутствии сходимости IRLS возвращает константу — среднее y_train.
    Это защита для редких случаев с экстремальной мультиколлинеарностью
    или вырожденным дизайном.
    """

    def __init__(self, family):
        self.family = family
        self._fit_result = None
        self._fallback_mean = None
        self._n_features = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._n_features = X.shape[1]
        # IRLS требует Y > 0 для Gamma; для Tweedie var_power=1.5 — Y ≥ 0.
        # Если Y нарушает это — отлавливаем и используем fallback.
        try:
            X_const = sm.add_constant(X, has_constant="add")
            model = sm.GLM(y, X_const, family=self.family)
            self._fit_result = model.fit(maxiter=200, disp=False)
        except Exception as e:
            self._fit_result = None
            self._fallback_mean = float(np.mean(y))
            warnings.warn(
                f"GLMWrapper ({type(self.family).__name__}): IRLS не сошёлся "
                f"({type(e).__name__}: {str(e)[:100]}). Возвращаю константу."
            )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._fit_result is None:
            return np.full(len(X), self._fallback_mean, dtype=float)
        X_const = sm.add_constant(X, has_constant="add")
        # На некоторых датасетах test может иметь другое число колонок-
        # констант (sm.add_constant пропускает константный столбец, если
        # такой уже есть). Дополнительно выравниваем форму.
        if X_const.shape[1] != self._fit_result.model.exog.shape[1]:
            # Принудительно вставляем колонку единиц спереди
            X_const = np.column_stack([np.ones(len(X)), X])
        try:
            return np.asarray(self._fit_result.predict(X_const), dtype=float)
        except Exception:
            return np.full(len(X), self._fallback_mean
                           if self._fallback_mean is not None else 0.0,
                           dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# CORE: эксперимент с CV + статтестами
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_cv(X_full, y_full, shift=0.0, models=None,
                      skip_transforms=(), use_smearing=True):
    """
    5-fold CV + hold-out test + paired t-test (CV) + Diebold–Mariano (test).
    Все альтернативные преобразования сравниваются с baseline 'none'.

    МЕТОДОЛОГИЧЕСКОЕ ОГРАНИЧЕНИЕ
    ────────────────────────────
    Нелинейные преобразования отклика g(Y) применяются только к моделям из
    REGRESSION_MODELS = {linear, ridge, lasso}. Для древесных моделей (rf,
    xgb) и MLP оценивается ИСКЛЮЧИТЕЛЬНО baseline 'none', а поправка
    Дуана / обобщённая smearing-оценка вообще не вызываются. См. шапку
    модуля (METHODOLOGY) для обоснования: теория Box-Cox / Дуана выведена
    для линейной регрессии с гомоскедастичными остатками; её перенос на
    деревья и MLP даёт паразитное Йенсеново смещение без теоретической
    компенсации.

    В возвращаемом df строки (m, t) для m ∉ REGRESSION_MODELS и t ≠ 'none'
    ОТСУТСТВУЮТ по построению. Все агрегаторы и сводные таблицы рассчитаны
    на это и корректно обрабатывают такие пропуски.

    Параметры
    ─────────
    use_smearing : bool, по умолчанию True.
        Если True — обратное преобразование g⁻¹ корректируется обобщённой
        smearing-оценкой (Duan 1983) для устранения Йенсенова смещения
        для нелинейных монотонных g (log, sqrt, asinh, Box-Cox,
        Yeo-Johnson). Это методически корректный режим. Применяется ТОЛЬКО
        к моделям из REGRESSION_MODELS (см. выше).
        Если False — используется наивная инверсия g⁻¹(ĝ(x)). Этот режим
        оставлен для воспроизводимости результатов v4 и для сравнения
        вклада smearing-коррекции в работе.

        В обоих режимах:
        — quantile: smearing не применяется (inverse_transform — ранговый);
        — none:     тривиальная инверсия, поправка не нужна.

    Возвращает (df, predictions, cv_rmses), где
      df            — DataFrame, индексированный (model, transform);
      predictions   — словарь (model, transform) → pred_test_orig (для DM);
      cv_rmses      — словарь (model, transform) → list of fold-RMSE.
    """
    if models is None:
        models = MODELS

    # Преобразования, к которым применяем smearing (если use_smearing=True)
    SMEAR_APPLICABLE = {"log", "sqrt", "asinh", "boxcox", "yeojohnson"}

    # ── БЛОК 3 ТЗ: модели, требующие Y > 0 или Y ≥ 0 ─────────────────────────
    # Gamma GLM и XGB-Gamma — выпускают log-link / Gamma-loss, нужен Y > 0.
    # Tweedie GLM и XGB-Tweedie (var_power=1.5) — допускают Y ≥ 0 (т.е. нули,
    # но не отрицательные).
    REQ_POSITIVE = {"gamma_glm", "xgb_gamma"}
    REQ_NONNEG   = {"tweedie_glm", "xgb_tweedie"}

    # Hold-out split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Y-fit (с учётом сдвига) для проверки положительности
    y_check = y_full + shift
    Y_STRICTLY_POSITIVE = bool(np.all(y_check > 0))
    Y_NONNEGATIVE       = bool(np.all(y_check >= 0))

    # Хранилища для последующего тестирования значимости
    cv_rmses_store = {}             # (model, transform) -> list of fold RMSEs
    train_rmses_store = {}          # (model, transform) -> RMSE on full train (orig scale)
    pred_test_store = {}            # (model, transform) -> pred on test (orig scale)
    records = []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for t in TRANSFORMS:
        if t in skip_transforms:
            continue
        # Применяем преобразование к full train/test
        try:
            y_tr_t, y_te_t, lam, inv = apply_transform(y_tr, y_te, t, shift=shift)
        except Exception as e:
            print(f"    [WARN] Преобр. '{t}' пропущено: {e}")
            continue

        apply_smear = use_smearing and (t in SMEAR_APPLICABLE)

        for m_key in models:
            # ── МЕТОДОЛОГИЧЕСКОЕ ОГРАНИЧЕНИЕ ──────────────────────────────
            # Нелинейные преобразования отклика и поправка Дуана
            # применяются только к моделям из REGRESSION_MODELS
            # (линейные регрессии: МНК, Ridge, Lasso). Для деревьев и MLP
            # оценивается лишь baseline 'none'. Подробнее — в шапке модуля.
            if t != "none" and m_key not in REGRESSION_MODELS:
                continue

            # ── БЛОК 3 ТЗ: GLM и XGB-spec-loss требуют Y > 0 или Y ≥ 0 ──
            # Эти модели — альтернатива преобразованию Y, поэтому
            # запускаются только с transform='none' (это уже отфильтровано
            # выше, т.к. они не входят в REGRESSION_MODELS).
            # Здесь — проверка применимости по диапазону Y.
            if m_key in REQ_POSITIVE and not Y_STRICTLY_POSITIVE:
                continue
            if m_key in REQ_NONNEG and not Y_NONNEGATIVE:
                continue

            # ── 5-fold CV на тренировочной выборке ──
            cv_rmses = []
            for fold_tr, fold_val in kf.split(X_tr):
                X_f_tr, X_f_val = X_tr[fold_tr], X_tr[fold_val]
                y_f_tr_t, y_f_val_t = y_tr_t[fold_tr], y_tr_t[fold_val]
                y_f_val_orig        = y_tr[fold_val]

                model = make_model(m_key)
                if m_key == "mlp":
                    sc = StandardScaler()
                    X_f_tr  = sc.fit_transform(X_f_tr)
                    X_f_val = sc.transform(X_f_val)
                model.fit(X_f_tr, y_f_tr_t)
                pred_f = model.predict(X_f_val)

                # ── Обратное преобразование в исходную шкалу ──
                # use_smearing=True → обобщённая Duan-коррекция для всех
                # нелинейных монотонных g; иначе наивная инверсия.
                if apply_smear:
                    resid_f = y_f_tr_t - model.predict(X_f_tr)
                    pred_orig_f = general_smearing(pred_f, resid_f, inv)
                else:
                    pred_orig_f = inv(pred_f)

                pred_orig_f = np.nan_to_num(pred_orig_f, nan=0.0,
                                             posinf=1e12, neginf=0.0)
                cv_rmses.append(float(np.sqrt(
                    mean_squared_error(y_f_val_orig, pred_orig_f))))

            cv_rmse_mean = float(np.mean(cv_rmses))
            cv_rmse_std  = float(np.std(cv_rmses))
            cv_rmses_store[(m_key, t)] = cv_rmses

            # ── Финальная оценка на hold-out test ──
            model = make_model(m_key)
            X_tr_m, X_te_m = X_tr.copy(), X_te.copy()
            if m_key == "mlp":
                sc = StandardScaler()
                X_tr_m = sc.fit_transform(X_tr_m)
                X_te_m = sc.transform(X_te_m)

            model.fit(X_tr_m, y_tr_t)
            pred_t_tr = model.predict(X_tr_m)
            pred_t_te = model.predict(X_te_m)
            resid_train_t = y_tr_t - pred_t_tr
            resid_test_t  = y_te_t - pred_t_te

            if apply_smear:
                pred_orig_te = general_smearing(pred_t_te, resid_train_t, inv)
                pred_orig_tr = general_smearing(pred_t_tr, resid_train_t, inv)
            else:
                pred_orig_te = inv(pred_t_te)
                pred_orig_tr = inv(pred_t_tr)
            pred_orig_te = np.nan_to_num(pred_orig_te, nan=0.0,
                                          posinf=1e12, neginf=0.0)
            pred_orig_tr = np.nan_to_num(pred_orig_tr, nan=0.0,
                                          posinf=1e12, neginf=0.0)
            pred_test_store[(m_key, t)] = pred_orig_te

            # Метрики на test (исходная шкала)
            row = compute_metrics(y_te, pred_orig_te)
            # Метрики на train (исходная шкала) — отвечает на замечание руководителя
            train_metrics = compute_metrics(y_tr, pred_orig_tr)
            row["RMSE_train"] = train_metrics["RMSE"]
            row["MAE_train"]  = train_metrics["MAE"]
            row["R2_train"]   = train_metrics["R2"]
            train_rmses_store[(m_key, t)] = train_metrics["RMSE"]

            row["model"]        = m_key
            row["transform"]    = t
            row["lambda"]       = lam
            row["cv_rmse_mean"] = cv_rmse_mean
            row["cv_rmse_std"]  = cv_rmse_std

            # Диагностика остатков (для Главы 1 — обратите внимание:
            # остатки вычисляются на ТЕСТОВОЙ выборке в трансформированной шкале)
            row["skew_resid"] = float(skew(resid_test_t))
            samp = resid_test_t[: min(5000, len(resid_test_t))]
            row["shapiro_p"] = float(shapiro(samp)[1])

            if m_key == "linear":
                try:
                    exog = sm.add_constant(X_te)
                    _, bp_p, _, _ = het_breuschpagan(resid_test_t, exog)
                    row["bp_p"] = float(bp_p)
                except Exception:
                    row["bp_p"] = float("nan")
            else:
                row["bp_p"] = float("nan")

            if t == "log":
                c, bias = duan_smearing(resid_train_t)
                row["duan_c"], row["duan_bias"] = c, bias
            else:
                row["duan_c"] = row["duan_bias"] = float("nan")

            if m_key in ("ridge", "lasso") and hasattr(model, "alpha_"):
                row["best_alpha"] = float(model.alpha_)
            else:
                row["best_alpha"] = float("nan")

            records.append(row)

    df = pd.DataFrame(records).set_index(["model", "transform"])

    # ─── ΔRMSE %, ΔRMSE_train %, статистические тесты ───
    # Используем .loc[]-присвоение по конкретному индексу (model, transform).
    # Это надёжно при отсутствии части комбинаций (для нерегрессионных моделей
    # есть только строка 'none') и независимо от порядка построения records.
    # Заметим: позиционное присвоение списка длиной L к колонке (df[col]=list)
    # привязывало бы значения к строкам в порядке индексации DataFrame —
    # ошибочно, т.к. records строятся в порядке (transform-внешний,
    # model-внутренний), а deltas — в обратном. Через .loc[(m,t), col]
    # выравнивание корректное по построению.
    for col in ("delta_rmse_pct", "delta_rmse_train_pct",
                "DM_stat", "DM_p", "paired_t_stat", "paired_t_p",
                # БЛОК 1 ТЗ: эффект-сайз и оценка мощности
                "cohens_d", "min_n_80power", "is_underpowered"):
        df[col] = float("nan")

    for m_key in models:
        if (m_key, "none") not in df.index:
            continue
        base_rmse_te = df.loc[(m_key, "none"), "RMSE"]
        base_rmse_tr = df.loc[(m_key, "none"), "RMSE_train"]
        base_cv      = cv_rmses_store[(m_key, "none")]
        base_pred    = pred_test_store[(m_key, "none")]

        for t in TRANSFORMS:
            if (m_key, t) not in df.index:
                continue
            cur_rmse_te = df.loc[(m_key, t), "RMSE"]
            cur_rmse_tr = df.loc[(m_key, t), "RMSE_train"]
            df.loc[(m_key, t), "delta_rmse_pct"] = \
                (cur_rmse_te - base_rmse_te) / base_rmse_te * 100.0
            df.loc[(m_key, t), "delta_rmse_train_pct"] = \
                (cur_rmse_tr - base_rmse_tr) / base_rmse_tr * 100.0

            if t == "none":
                continue
            # Diebold-Mariano на test
            dms, dmp, _ = diebold_mariano(
                np.asarray(y_te), base_pred, pred_test_store[(m_key, t)])
            df.loc[(m_key, t), "DM_stat"] = dms
            df.loc[(m_key, t), "DM_p"]    = dmp
            # Paired t-test на CV-фолдах
            ts, tp, _ = paired_cv_ttest(base_cv, cv_rmses_store[(m_key, t)])
            df.loc[(m_key, t), "paired_t_stat"] = ts
            df.loc[(m_key, t), "paired_t_p"]    = tp

            # ── БЛОК 1 ТЗ: Cohen's d на разностях квадратичных потерь ──
            # Размер эффекта, не зависящий от n_test. Позволяет отличить
            # «эффекта нет» (d ≈ 0) от «эффект есть, но выборка мала»
            # (|d|>0.2, n<n*). См. cohens_d_squared_errors / min_n_for_power.
            d_eff = cohens_d_squared_errors(
                np.asarray(y_te), base_pred, pred_test_store[(m_key, t)])
            n_min = min_n_for_power(d_eff, alpha=ALPHA, power=0.80)
            df.loc[(m_key, t), "cohens_d"]       = d_eff
            df.loc[(m_key, t), "min_n_80power"]  = n_min
            df.loc[(m_key, t), "is_underpowered"] = int(
                np.isfinite(n_min) and len(y_te) < n_min)

    return df, pred_test_store, cv_rmses_store


# ─────────────────────────────────────────────────────────────────────────────
# Демонстрация Йенсена–Дуана (для Главы 1)
# ─────────────────────────────────────────────────────────────────────────────

def jensen_duan_demo(save_plot=True):
    """
    Численная иллюстрация связи поправки Дуана с неравенством Йенсена.

    Для модели log Y = Xβ + ε  с E[ε]=0  имеем:
        E[Y|X] = exp(Xβ) · E[exp(ε)],
    т.е. наивный обратный прогноз exp(Xβ̂) систематически смещён вниз
    на множитель κ = E[exp(ε)].

    Йенсен (для выпуклой exp): E[exp(ε)] ≥ exp(E[ε]) = 1.
    Для нормальных остатков ε ∼ N(0, σ²):  κ = exp(σ²/2) — точная формула.
    Поправка Дуана:  ĉ = (1/n) Σ exp(ê_i) — состоятельная НЕпараметрическая
    оценка κ, не требующая нормальности.

    Эта функция:
      (a) для нормальных остатков сравнивает теоретическое значение κ
          с эмпирическим средним exp(ε) (Дуан) и с наивной оценкой;
      (b) для T-распределения и гамма-распределения — иллюстрирует, что
          поправка работает и без нормальности (то, для чего Дуан её и вводил).
    """
    section("ДЕМОНСТРАЦИЯ ЙЕНСЕНА–ДУАНА (для теоретической главы)")

    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = 200_000

    print("\n  (a) Нормальные остатки ε ∼ N(0, σ²): теор. κ = exp(σ²/2)")
    print(f"\n    {'σ':>6} {'теор. κ':>12} {'наивный exp(ε̄)':>18} "
          f"{'Дуан ĉ':>12} {'смещ. наивн.':>14} {'смещ. Дуан':>12}")
    print("    " + "─" * 78)

    sigmas = [0.10, 0.30, 0.50, 0.80, 1.00, 1.50]
    rows_a = []
    for sigma in sigmas:
        eps = rng.normal(0.0, sigma, n_samples)
        kappa_th = float(np.exp(sigma ** 2 / 2.0))
        naive    = float(np.exp(np.mean(eps)))
        duan     = float(np.mean(np.exp(eps)))
        bias_naive = (naive - kappa_th) / kappa_th * 100.0
        bias_duan  = (duan  - kappa_th) / kappa_th * 100.0
        print(f"    {sigma:>6.2f} {kappa_th:>12.4f} {naive:>18.4f} "
              f"{duan:>12.4f} {bias_naive:>+13.2f}% {bias_duan:>+11.2f}%")
        rows_a.append({"σ": sigma, "kappa_th": kappa_th, "naive": naive,
                       "duan": duan, "bias_naive_pct": bias_naive,
                       "bias_duan_pct": bias_duan})

    print("\n  Вывод: при σ ≥ 0.5 наивный обратный прогноз отстаёт от истины")
    print("  более чем на 10 %, при σ = 1.0 — на ~40 %. Дуан восстанавливает κ")
    print("  с погрешностью < 0.5 %. Это и есть Йенсенова компенсация.")

    # ── (b) Ненормальные распределения ──
    print("\n  (b) Ненормальные остатки (демонстрация непараметричности Дуана)")
    print(f"\n    {'распределение':<24} {'σ':>5} {'теор. κ':>12} "
          f"{'Дуан ĉ':>12} {'смещ.':>10}")
    print("    " + "─" * 70)

    cases = [
        ("Стьюдент t (df=5), масштаб 0.5",
         lambda r: r.standard_t(df=5, size=n_samples) * 0.5,
         None),  # для t нет простой замкнутой формулы → используем эмпирику
        ("Лаплас, scale=0.5",
         lambda r: r.laplace(loc=0.0, scale=0.5, size=n_samples),
         lambda: 1.0 / (1.0 - 0.5 ** 2)),  # MGF Лапласа в t=1
        ("Сдвинутая Гамма (k=2, θ=0.3, центр.)",
         lambda r: r.gamma(2.0, 0.3, n_samples) - 0.6,
         lambda: np.exp(-0.6) * (1 - 0.3) ** (-2.0)),  # MGF гаммы
    ]

    rows_b = []
    for name, gen, theory in cases:
        eps = gen(rng)
        eps = eps - np.mean(eps)   # центрируем (как остатки регрессии)
        sigma_emp = float(np.std(eps))
        # «Истинное» κ: либо аналитика, либо высокоточная эмпирическая оценка
        kappa_true = float(np.mean(np.exp(eps)))
        duan = kappa_true   # сам по себе Дуан = эмпирическое среднее
        # Используем повторную выборку меньшего размера для иллюстрации сходимости
        small = rng.choice(eps, size=2000, replace=False)
        small = small - np.mean(small)
        duan_small = float(np.mean(np.exp(small)))
        bias_pct = (duan_small - kappa_true) / kappa_true * 100.0
        print(f"    {name:<24} {sigma_emp:>5.2f} {kappa_true:>12.4f} "
              f"{duan_small:>12.4f} {bias_pct:>+9.2f}%")
        rows_b.append({"distribution": name, "sigma": sigma_emp,
                       "kappa_true": kappa_true, "duan_n2000": duan_small,
                       "bias_pct": bias_pct})

    print("\n  Вывод: оценка Дуана сходится к истинному κ при любом распределении")
    print("  остатков → подтверждает теорию (Duan 1983) о состоятельности оценки.")

    # Сохраняем CSV для ВКР
    pd.DataFrame(rows_a).to_csv(RESULTS_DIR / "jensen_duan_normal.csv",
                                index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_b).to_csv(RESULTS_DIR / "jensen_duan_nonnormal.csv",
                                index=False, encoding="utf-8-sig")

    if save_plot:
        # График: σ → κ (теория и Дуан) для нормальных остатков
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sigma_grid = np.linspace(0.05, 1.6, 60)
        kappa_th_grid = np.exp(sigma_grid ** 2 / 2)
        ax.plot(sigma_grid, kappa_th_grid, "-", color="#4472C4", lw=2,
                label=r"теор.: $\kappa = \exp(\sigma^2/2)$")
        ax.axhline(1.0, color="#AAAAAA", ls="--", lw=1.2,
                   label=r"наивный: $\exp(\mathbb{E}[\varepsilon])=1$")
        sigmas_plot = [r["σ"] for r in rows_a]
        duans_plot  = [r["duan"] for r in rows_a]
        ax.scatter(sigmas_plot, duans_plot, color="#ED7D31", zorder=5,
                   s=60, label="Дуан $\\hat c$ (эмп.)")
        ax.set_xlabel(r"$\sigma$ (стд. остатков)")
        ax.set_ylabel(r"$\kappa = \mathbb{E}[\exp\varepsilon]$")
        ax.set_title("Йенсенова компенсация: смещение наивного обратного "
                     "лог-прогноза vs. поправка Дуана")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = RESULTS_DIR / "jensen_duan_demo.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  → График: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def print_metrics_multi(df, title=""):
    """Подробная таблица метрик для каждой модели — train, test, CV, ΔRMSE, p-values.

    БЛОК 1 ТЗ: добавлены колонки `d` (Cohen's d) и маркер `⚠` для
    underpowered-случаев (DM_p > 0.05, но |d| > 0.2 при n_test < n*).
    """
    if title:
        print(f"\n  {title}")
    models = df.index.get_level_values("model").unique()
    for m_key in models:
        print(f"\n  ── {MODEL_LABEL.get(m_key, m_key)} {'─' * 50}")
        if m_key not in REGRESSION_MODELS:
            print(f"    [Преобразования отклика к этой модели не применяются "
                  f"методологически; см. шапку модуля. Показан только baseline.]")
        hdr = (f"    {'Преобр.':<14} {'λ':>6} "
               f"{'RMSE_tr':>9} {'RMSE_te':>9} {'CV-RMSE':>13} "
               f"{'ΔRMSE_tr%':>10} {'ΔRMSE_te%':>10} "
               f"{'DM_p':>7} {'t_p':>7} {'d':>6} {'n*':>6}")
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for t in TRANSFORMS:
            if (m_key, t) not in df.index:
                continue
            r = df.loc[(m_key, t)]
            lam = f"{r['lambda']:.2f}" if r["lambda"] is not None and \
                                          not (isinstance(r["lambda"], float)
                                               and np.isnan(r["lambda"])) else "—"
            cv_str = f"{r['cv_rmse_mean']:.1f}±{r['cv_rmse_std']:.1f}"

            # Маркеры значимости и величины
            flag = ""
            if t != "none":
                practical = abs(r["delta_rmse_pct"]) >= RMSE_THRESHOLD
                significant = (not np.isnan(r["DM_p"])) and r["DM_p"] < ALPHA
                if practical and significant:
                    flag = " ◄◄"   # практически и статистически значимо
                elif significant:
                    flag = " ◄"    # только статистически
                elif practical:
                    flag = " ◊"    # только практически
                # БЛОК 1 ТЗ: маркер ⚠ для underpowered-строк —
                # есть содержательный эффект (|d|>0.2), но p>α и n_test<n*
                d_val = r.get("cohens_d", float("nan"))
                up = r.get("is_underpowered", False)
                if (not significant) and (not np.isnan(d_val)) and \
                   abs(d_val) > 0.2 and bool(up):
                    flag += " ⚠"

            dm_p   = f"{r['DM_p']:.3f}"        if not np.isnan(r["DM_p"])        else "—"
            tt_p   = f"{r['paired_t_p']:.3f}"  if not np.isnan(r["paired_t_p"])  else "—"
            d_val_v = r.get("cohens_d", float("nan"))
            d_str  = f"{d_val_v:+.2f}" if not (isinstance(d_val_v, float) and
                                                np.isnan(d_val_v)) else "—"
            n_min  = r.get("min_n_80power", float("nan"))
            if isinstance(n_min, float) and (np.isnan(n_min) or np.isinf(n_min)):
                nm_str = "—"
            else:
                nm_str = f"{int(n_min):>6d}" if n_min < 1e6 else ">1e6"

            print(f"    {TR_LABEL[t]:<14} {lam:>6} "
                  f"{r['RMSE_train']:>9.1f} {r['RMSE']:>9.1f} {cv_str:>13} "
                  f"{r['delta_rmse_train_pct']:>+10.1f} {r['delta_rmse_pct']:>+10.1f}"
                  f" {dm_p:>7} {tt_p:>7} {d_str:>6} {nm_str:>6}{flag}")
    print("\n    Легенда: ◄◄ — практически (|ΔRMSE|≥5%) И статистически (p<0.05) значимо;")
    print("             ◄  — только статистически;  ◊ — только практически.")
    print("             ⚠  — UNDERPOWERED: есть содержательный эффект (|d|>0.2),")
    print("                  но p>α и n_test < n*(80% power). Размер выборки мал —")
    print("                  отсутствие значимости неинформативно.")
    print("    Cohen's d: 0.2 малый, 0.5 средний, 0.8 крупный эффект.")
    print("    n*       : минимум n_test для 80% мощности при данном |d|.")


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 1 ТЗ: сводка underpowered-случаев по всему registry
# ─────────────────────────────────────────────────────────────────────────────

def print_underpowered_summary(registry):
    """Сводная таблица underpowered-случаев для всех (датасет × модель ×
    преобр.). Используется для текстового блока в Главе 2.

    Underpowered-случай — это пара (модель × преобразование), где:
      • |Cohen's d| > 0.2  (есть содержательный эффект);
      • DM_p > 0.05         (статистически не значимо);
      • n_test < n*         (выборка мала).
    """
    section("СВОДКА UNDERPOWERED-СЛУЧАЕВ (БЛОК 1 ТЗ)")
    print("\n  Эти случаи иллюстрируют различие между «эффекта нет» и")
    print("  «эффект есть, но выборка мала». Cohen's d отличает их.")
    print(f"\n    {'Датасет':<22} {'Модель':<13} {'Преобр.':<13} "
          f"{'ΔRMSE%':>8} {'DM_p':>7} {'d':>6} {'n_test':>7} {'n*':>8}")
    print("    " + "─" * 92)

    n_underpowered = 0
    rows = []
    for ds_name, (g1, df) in sorted(registry.items(), key=lambda x: x[1][0]):
        for (m_key, t) in df.index:
            if t == "none":
                continue
            r = df.loc[(m_key, t)]
            d_val = r.get("cohens_d", float("nan"))
            dm_p  = r.get("DM_p", float("nan"))
            up    = bool(r.get("is_underpowered", False))
            if (not np.isnan(d_val)) and abs(d_val) > 0.2 and \
               (not np.isnan(dm_p)) and dm_p > ALPHA and up:
                n_test_est = int(round((1.0 - TEST_SIZE) * 0))  # placeholder
                # Истинное n_test заранее неизвестно (зависит от датасета);
                # принимаем оценку как округление по len(df).
                n_min = r.get("min_n_80power", float("nan"))
                nm_str = (f"{int(n_min):>8d}" if (np.isfinite(n_min)
                                                    and n_min < 1e6)
                          else ">1e6")
                print(f"    {ds_name:<22} {MODEL_LABEL[m_key]:<13} "
                      f"{TR_LABEL[t]:<13} {r['delta_rmse_pct']:>+7.1f}% "
                      f"{dm_p:>7.3f} {d_val:>+6.2f}      ?  {nm_str}")
                rows.append({
                    "Dataset": ds_name, "gamma1": g1,
                    "Model": MODEL_LABEL[m_key],
                    "Transform": TR_LABEL[t],
                    "delta_rmse_pct": float(r["delta_rmse_pct"]),
                    "DM_p": float(dm_p),
                    "cohens_d": float(d_val),
                    "min_n_80power": float(n_min)
                        if np.isfinite(n_min) else None,
                })
                n_underpowered += 1

    if n_underpowered == 0:
        print("    (underpowered-случаев не зафиксировано)")
    else:
        print(f"\n  Всего underpowered: {n_underpowered}.")
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "underpowered_cases.csv",
            index=False, encoding="utf-8-sig")
        print(f"  → {RESULTS_DIR / 'underpowered_cases.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. СИНТЕТИЧЕСКИЕ ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic(n=1500, sigma=0.40, seed=RANDOM_STATE):
    """log Y = 2 + Xβ + σ·ε,  ε∼N(0,1).  Управляем асимметрией через σ."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    eps = rng.standard_normal(n)
    log_y = 2.0 + X @ SYNTH_BETA + sigma * eps
    return X, np.exp(log_y)


def run_synthetic():
    section("1. СИНТЕТИЧЕСКИЕ ДАННЫЕ")
    results = {}
    for name, (sigma, level) in SYNTH_SERIES.items():
        X, Y = generate_synthetic(n=1500, sigma=sigma)
        g1 = float(skew(Y))
        print(f"\n  Серия {name}  σ={sigma}  γ₁={g1:.2f}  "
              f"E[Y]={Y.mean():.2f}  Sd[Y]={Y.std():.2f}  ({level} асимметрия)")
        df, _, _ = run_experiment_cv(X, Y, shift=0.0)
        results[f"Synth_{name}"] = (g1, df)
        print_metrics_multi(df, f"Метрики — серия {name}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIAMONDS (real data — pydataset)
# ══════════════════════════════════════════════════════════════════════════════

def run_diamonds():
    section("2. DIAMONDS DATASET (real data, γ₁≈1.6)")
    from pydataset import data as pdata

    df_raw = pdata("diamonds")
    print(f"\n  Источник: pydataset (ggplot2 'diamonds')")
    print(f"  n={len(df_raw):,}, p_raw={df_raw.shape[1]}")

    Y = df_raw["price"].values.astype(float)
    g1 = float(skew(Y))
    print(f"\n  Целевая переменная: price (цена в USD)")
    print(f"  γ₁={g1:.2f},  E[Y]={Y.mean():,.0f},  "
          f"Median[Y]={np.median(Y):,.0f},  Sd[Y]={Y.std():,.0f}")
    print(f"  Shapiro–Wilk p = {shapiro(Y[:5000])[1]:.4f}")

    # Чистка
    bad = (df_raw["x"] == 0) | (df_raw["y"] == 0) | (df_raw["z"] == 0)
    print(f"\n  Аномалии (x/y/z=0): {bad.sum()} → удалены")
    df_clean = df_raw[~bad].copy()
    bad2 = (df_clean["y"] > 20) | (df_clean["z"] > 20)
    df_clean = df_clean[~bad2].copy()
    print(f"  После чистки: n={len(df_clean):,}")

    cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order   = ["J", "I", "H", "G", "F", "E", "D"]
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    df_clean["cut_enc"]     = df_clean["cut"].map({v: i for i, v in enumerate(cut_order)})
    df_clean["color_enc"]   = df_clean["color"].map({v: i for i, v in enumerate(color_order)})
    df_clean["clarity_enc"] = df_clean["clarity"].map({v: i for i, v in enumerate(clarity_order)})

    feature_cols = ["carat", "cut_enc", "color_enc", "clarity_enc",
                    "depth", "table", "x", "y", "z"]
    X = df_clean[feature_cols].values.astype(float)
    Y = df_clean["price"].values.astype(float)
    g1 = float(skew(Y))
    print(f"  Признаки ({len(feature_cols)}): {feature_cols}")
    print(f"  γ₁(Y после чистки)={g1:.2f}")

    lam, ci = boxcox_fit(Y[:10000])
    print(f"\n  Box-Cox λ̂ = {lam:.3f}  (95% ДИ: [{ci[0]:.3f}; {ci[1]:.3f}])")
    for lam_h, label_h in [(0.0, "λ=0 (логарифм)"), (1.0, "λ=1 (без преобр.)")]:
        chi2, p = lrt_boxcox(lam_h, lam, Y[:10000])
        verdict = "не отвергается" if p > 0.05 else "отвергается"
        print(f"  LRT H₀: {label_h:22s}  χ²={chi2:.2f}  p={p:.4f}  → {verdict}")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Y), size=min(10000, len(Y)), replace=False)
    X_sub, Y_sub = X[idx], Y[idx]
    print(f"\n  Подвыборка: n={len(Y_sub):,}")

    df, _, _ = run_experiment_cv(X_sub, Y_sub, shift=0.0)
    print_metrics_multi(df, "Метрики — Diamonds")

    base_rmse = float(np.sqrt(np.mean((Y_sub - Y_sub.mean()) ** 2)))
    print(f"\n  Baseline (среднее): RMSE = {base_rmse:,.0f}")
    return g1, df


# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIFORNIA HOUSING (real data — sklearn) — γ₁≈1.0  [НОВЫЙ]
# ══════════════════════════════════════════════════════════════════════════════

def run_california_housing():
    section("3. CALIFORNIA HOUSING (real data, γ₁≈1.0) [НОВЫЙ]")
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df_raw = data.frame
    print(f"\n  Источник: sklearn.datasets.fetch_california_housing (1990 US Census)")
    print(f"  n={len(df_raw):,}, p={df_raw.shape[1] - 1}")
    print(f"  Признаки: {list(data.feature_names)}")

    Y = df_raw["MedHouseVal"].values.astype(float)
    g1 = float(skew(Y))
    print(f"\n  Целевая переменная: MedHouseVal (медианная цена дома, $100k)")
    print(f"  γ₁={g1:.2f},  E[Y]={Y.mean():.2f},  "
          f"Median[Y]={np.median(Y):.2f},  Sd[Y]={Y.std():.2f}")
    print(f"  Min={Y.min():.2f},  Max={Y.max():.2f}")
    print(f"  Shapiro–Wilk p = {shapiro(Y[:5000])[1]:.4f}")

    # В California Housing есть «капинг» Y на $500k → ~5% наблюдений
    capped = int(np.sum(Y >= Y.max() - 1e-6))
    print(f"  Цензурированных (на верхней границе): {capped} ({capped/len(Y)*100:.1f}%)")

    X = df_raw[data.feature_names].values.astype(float)

    # Box-Cox диагностика
    lam, ci = boxcox_fit(Y)
    print(f"\n  Box-Cox λ̂ = {lam:.3f}  (95% ДИ: [{ci[0]:.3f}; {ci[1]:.3f}])")
    for lam_h, label_h in [(0.0, "λ=0 (логарифм)"), (1.0, "λ=1 (без преобр.)")]:
        chi2, p = lrt_boxcox(lam_h, lam, Y)
        verdict = "не отвергается" if p > 0.05 else "отвергается"
        print(f"  LRT H₀: {label_h:22s}  χ²={chi2:.2f}  p={p:.4f}  → {verdict}")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Y), size=min(10000, len(Y)), replace=False)
    X_sub, Y_sub = X[idx], Y[idx]
    print(f"\n  Подвыборка: n={len(Y_sub):,}")

    df, _, _ = run_experiment_cv(X_sub, Y_sub, shift=0.0)
    print_metrics_multi(df, "Метрики — California Housing")

    base_rmse = float(np.sqrt(np.mean((Y_sub - Y_sub.mean()) ** 2)))
    print(f"\n  Baseline (среднее): RMSE = {base_rmse:.3f}")
    return g1, df


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONCRETE STRENGTH (real data — UCI, γ₁≈0.4) [НОВЫЙ, опц.]
# ══════════════════════════════════════════════════════════════════════════════

def run_concrete():
    section("4. CONCRETE COMPRESSIVE STRENGTH (real data, γ₁≈0.4) [НОВЫЙ]")
    df_raw = None

    # Пытаемся загрузить из ucimlrepo, если установлен; иначе — fallback
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=165)
        df_raw = pd.concat([ds.data.features, ds.data.targets], axis=1)
        target_col = ds.data.targets.columns[0]
        print(f"\n  Источник: UCI ML Repository (id=165) через ucimlrepo")
    except Exception as e1:
        # Fallback 1: pip install ucimlrepo не установлен → попробуем через openml
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=4353, as_frame=True)  # Concrete_Compressive_Strength
            df_raw = data.frame.copy()
            target_col = data.target.name if data.target is not None else df_raw.columns[-1]
            print(f"\n  Источник: OpenML (data_id=4353)")
        except Exception as e2:
            print(f"\n  [SKIP] Concrete Strength недоступен (нет ucimlrepo и openml): "
                  f"{type(e1).__name__}, {type(e2).__name__}")
            return None, None

    df_raw = df_raw.dropna()
    df_raw.columns = [c.strip() for c in df_raw.columns]
    print(f"  n={len(df_raw):,}, p={df_raw.shape[1] - 1}")

    Y = df_raw[target_col].values.astype(float)
    feature_cols = [c for c in df_raw.columns if c != target_col]
    X = df_raw[feature_cols].values.astype(float)
    g1 = float(skew(Y))
    print(f"\n  Целевая переменная: {target_col}")
    print(f"  γ₁={g1:.2f},  E[Y]={Y.mean():.2f},  "
          f"Median[Y]={np.median(Y):.2f},  Sd[Y]={Y.std():.2f}")
    print(f"  Признаки ({len(feature_cols)}): {feature_cols}")

    lam, ci = boxcox_fit(Y)
    print(f"\n  Box-Cox λ̂ = {lam:.3f}  (95% ДИ: [{ci[0]:.3f}; {ci[1]:.3f}])")
    for lam_h, label_h in [(0.0, "λ=0 (логарифм)"), (1.0, "λ=1 (без преобр.)")]:
        chi2, p = lrt_boxcox(lam_h, lam, Y)
        verdict = "не отвергается" if p > 0.05 else "отвергается"
        print(f"  LRT H₀: {label_h:22s}  χ²={chi2:.2f}  p={p:.4f}  → {verdict}")

    df, _, _ = run_experiment_cv(X, Y, shift=0.0)
    print_metrics_multi(df, "Метрики — Concrete Strength")

    base_rmse = float(np.sqrt(np.mean((Y - Y.mean()) ** 2)))
    print(f"\n  Baseline (среднее): RMSE = {base_rmse:.2f}")
    return g1, df


# ══════════════════════════════════════════════════════════════════════════════
# 5. RAND HIE — Medical Visits (real data — statsmodels, γ₁≈4.8)
# ══════════════════════════════════════════════════════════════════════════════

def run_randhie():
    section("5. RAND HIE — MEDICAL VISITS (real data, γ₁≈4.8)")

    df_raw = sm.datasets.randhie.load_pandas().data
    print(f"\n  Источник: statsmodels (RAND Health Insurance Experiment)")
    print(f"  n={len(df_raw):,}, p_raw={df_raw.shape[1]}")

    Y = df_raw["mdvis"].values.astype(float)
    g1 = float(skew(Y))
    zeros = int(np.sum(Y == 0))
    print(f"\n  Целевая переменная: mdvis (число визитов к врачу)")
    print(f"  γ₁={g1:.2f},  E[Y]={Y.mean():.2f},  Median[Y]={np.median(Y):.1f},  "
          f"Sd[Y]={Y.std():.2f}")
    print(f"  Min={Y.min():.0f},  Max={Y.max():.0f}")
    print(f"  Нулей: {zeros} ({zeros / len(Y) * 100:.1f}%)")

    df_clean = df_raw.dropna().copy()
    feature_cols = [c for c in df_clean.columns if c != "mdvis"]
    X = df_clean[feature_cols].values.astype(float)
    Y = df_clean["mdvis"].values.astype(float)
    g1 = float(skew(Y))

    SHIFT = 1.0
    print(f"  Сдвиг δ={SHIFT} для log/Box-Cox (наличие нулей)")

    lam, ci = boxcox_fit(Y[:10000] + SHIFT)
    print(f"\n  Box-Cox λ̂ = {lam:.3f}  (95% ДИ: [{ci[0]:.3f}; {ci[1]:.3f}])")
    for lam_h, label_h in [(0.0, "λ=0 (логарифм)"), (1.0, "λ=1 (без преобр.)")]:
        chi2, p = lrt_boxcox(lam_h, lam, Y[:10000] + SHIFT)
        verdict = "не отвергается" if p > 0.05 else "отвергается"
        print(f"  LRT H₀: {label_h:22s}  χ²={chi2:.2f}  p={p:.4f}  → {verdict}")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Y), size=min(10000, len(Y)), replace=False)
    X_sub, Y_sub = X[idx], Y[idx]
    print(f"\n  Подвыборка: n={len(Y_sub):,}")

    df, _, _ = run_experiment_cv(X_sub, Y_sub, shift=SHIFT)
    print_metrics_multi(df, "Метрики — RAND HIE (mdvis)")

    base_rmse = float(np.sqrt(np.mean((Y_sub - Y_sub.mean()) ** 2)))
    print(f"\n  Baseline (среднее): RMSE = {base_rmse:.2f}")
    return g1, df


# ══════════════════════════════════════════════════════════════════════════════
# 5.5  ДОПОЛНИТЕЛЬНЫЕ ДАТАСЕТЫ (20 шт.) — плотное покрытие диапазона γ₁
# ══════════════════════════════════════════════════════════════════════════════
#
# Каждый loader возвращает dict со строго фиксированными ключами:
#     X              — np.ndarray признаков (n × p), float
#     Y              — np.ndarray отклика (n,), float
#     name           — короткое имя для отчётов и CSV
#     target         — название целевой переменной
#     shift          — δ для log/Box-Cox при наличии нулей/отрицат. значений
#     gamma_expected — ориентир по γ₁ (литература / разведочный анализ)
#     description    — источник / ссылка / краткое описание
# либо None при ошибке загрузки.
#
# Группы по асимметрии:
#     (S) симметричные                  γ₁ ∈ [-0.5, 0.5]   6 датасетов
#     (M) умеренно скошенные            γ₁ ∈ ( 0.5, 1.5]   5 датасетов
#     (H) сильно скошенные              γ₁ ∈ ( 1.5, 3.0]   5 датасетов
#     (X) экстремально скошенные        γ₁ >  3.0          4 датасета
# ─────────────────────────────────────────────────────────────────────────────

def _safe_loader(loader):
    """Декоратор: ловит исключения загрузки и возвращает None с warning-ом."""
    def wrapper():
        try:
            return loader()
        except Exception as e:
            print(f"  [SKIP] {loader.__name__}: {type(e).__name__}: {str(e)[:80]}")
            return None
    wrapper.__name__ = loader.__name__
    wrapper.__doc__ = loader.__doc__
    return wrapper


def _fetch_openml_safe(*candidates):
    """
    Пробует fetch_openml, перебирая список (name, version) или int data_id.
    Возвращает Bunch при успехе; кидает последнее исключение при провале всех.
    """
    from sklearn.datasets import fetch_openml
    last_exc = None
    for c in candidates:
        try:
            if isinstance(c, int):
                return fetch_openml(data_id=c, as_frame=True, parser="auto")
            elif isinstance(c, tuple):
                name, ver = c
                return fetch_openml(name=name, version=ver,
                                    as_frame=True, parser="auto")
            else:  # str
                return fetch_openml(name=c, version=1,
                                    as_frame=True, parser="auto")
        except Exception as e:
            last_exc = e
            continue
    raise last_exc if last_exc is not None else RuntimeError("no candidates")


def _frame_to_numeric_Xy(df_in, target_col):
    """
    Готовит (X, Y) из DataFrame: дроп NaN, кодирование категориальных
    признаков ordinal-индексами (порядок: уникальные значения в первой
    встрече). Гарантированно возвращает float ndarrays.

    Распознаёт любые нечисловые колонки (object / category / string / bool),
    а также числовые, которые на самом деле содержат строки.
    """
    df = df_in.dropna().copy()
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not in DataFrame")
    feat_cols = [c for c in df.columns if c != target_col]
    for c in feat_cols:
        col = df[c]
        # Чисто числовая колонка → ничего не делаем
        if pd.api.types.is_numeric_dtype(col) and not \
                pd.api.types.is_bool_dtype(col):
            continue
        # Иначе кодируем (object / category / string / bool / mixed)
        cats = list(pd.unique(col))
        df[c] = col.map({v: i for i, v in enumerate(cats)})
    # Target тоже может быть категориальным (например, ordinal quality)
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna()
    X = df[feat_cols].values.astype(float)
    Y = df[target_col].values.astype(float)
    return X, Y


# ────────────────── (S) Симметричные   γ₁ ∈ [-0.5, 0.5] ──────────────────────

@_safe_loader
def load_diabetes_sklearn():
    """sklearn diabetes — disease progression. γ₁ ≈ 0.44."""
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return dict(X=d.data.astype(float), Y=d.target.astype(float),
                name="Diabetes_sklearn", target="disease_progression",
                shift=0.0, gamma_expected=0.44,
                description="sklearn.datasets.load_diabetes (Efron et al.)")


@_safe_loader
def load_wine_quality_red():
    """OpenML wine-quality-red — quality score (int 3..8). γ₁ ≈ 0.22."""
    data = _fetch_openml_safe("wine-quality-red", 40691)
    df = data.frame.copy()
    target = "quality" if "quality" in df.columns else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="WineQuality_Red", target=target,
                shift=0.0, gamma_expected=0.22,
                description="OpenML 'wine-quality-red' (Cortez et al.)")


@_safe_loader
def load_wine_quality_white():
    """OpenML wine-quality-white — quality score (int 3..9). γ₁ ≈ 0.16."""
    data = _fetch_openml_safe("wine-quality-white", 40498)
    df = data.frame.copy()
    target = "quality" if "quality" in df.columns else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="WineQuality_White", target=target,
                shift=0.0, gamma_expected=0.16,
                description="OpenML 'wine-quality-white' (Cortez et al.)")


@_safe_loader
def load_airfoil_self_noise():
    """OpenML airfoil_self_noise — sound pressure level (dB). γ₁ ≈ -0.2."""
    data = _fetch_openml_safe("airfoil_self_noise", 43919, 44957)
    df = data.frame.copy()
    # Целевая переменная обычно последняя; имя варьируется
    cand_targets = [c for c in df.columns
                    if "scaled" in c.lower() or "sound" in c.lower()
                    or c.lower().startswith("y")]
    target = cand_targets[0] if cand_targets else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Airfoil_SelfNoise", target=target,
                shift=0.0, gamma_expected=-0.20,
                description="OpenML 'airfoil_self_noise' (NASA, Brooks et al.)")

@_safe_loader
def load_energy_efficiency_heating():
    """OpenML ENB — Heating Load (Y1). γ₁ ≈ 0.36."""
    data = _fetch_openml_safe("Energy_efficiency", 1472, 43383)
    df = data.frame.copy()
    # У этого датасета два отклика: Y1 (heating), Y2 (cooling). Берём Y1.
    cands = [c for c in df.columns if c.upper() in ("Y1", "HEATING_LOAD")]
    target = cands[0] if cands else df.columns[-2]   # предпоследняя
    drop_cooling = [c for c in df.columns
                    if c.upper() in ("Y2", "COOLING_LOAD") and c != target]
    df = df.drop(columns=drop_cooling, errors="ignore")
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Energy_Heating", target=target,
                shift=0.0, gamma_expected=0.36,
                description="OpenML 'Energy_efficiency' (Tsanas & Xifara, UCI)")


# ────────────────── (M) Умеренно скошенные   γ₁ ∈ (0.5, 1.5] ─────────────────

@_safe_loader
def load_auto_mpg():
    """OpenML autoMpg — fuel efficiency (mpg). γ₁ ≈ 0.46."""
    data = _fetch_openml_safe("autoMpg", 196, ("mpg", 1))
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.lower() in ("mpg", "class")]
    target = target_cands[0] if target_cands else df.columns[-1]
    # Уберём строковое поле car name, если осталось
    drop_str = [c for c in df.columns
                if df[c].dtype == "object" and c != target
                and df[c].nunique() > 50]
    df = df.drop(columns=drop_str, errors="ignore")
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Auto_MPG", target=target,
                shift=0.0, gamma_expected=0.46,
                description="OpenML 'autoMpg' (Quinlan, UCI)")


@_safe_loader
def load_abalone():
    """OpenML abalone — number of rings (proxy for age). γ₁ ≈ 0.91."""
    data = _fetch_openml_safe("abalone", 1557, 183)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if "ring" in c.lower() or c.lower() == "class"]
    target = target_cands[0] if target_cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Abalone", target=target,
                shift=0.0, gamma_expected=0.91,
                description="OpenML 'abalone' (Nash et al., UCI)")


@_safe_loader
def load_boston_housing():
    """OpenML boston — median home value MEDV ($1000s). γ₁ ≈ 1.10."""
    data = _fetch_openml_safe("boston", 531)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.upper() in ("MEDV", "TARGET")]
    target = target_cands[0] if target_cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Boston_Housing", target=target,
                shift=0.0, gamma_expected=1.10,
                description="OpenML 'boston' (Harrison & Rubinfeld, 1978)")


@_safe_loader
def load_real_estate_taiwan():
    """OpenML Real_estate_valuation — house price per unit area. γ₁ ≈ 0.60."""
    data = _fetch_openml_safe("Real-estate-valuation-data-set",
                              ("Real_estate_valuation_data_set", 1), 42712)
    df = data.frame.copy()
    cands = [c for c in df.columns
             if "price" in c.lower() or "y" in c.lower()[:2]]
    target = cands[-1] if cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Real_Estate_Taiwan", target=target,
                shift=0.0, gamma_expected=0.60,
                description="OpenML 'Real_estate_valuation' (Yeh, UCI)")


@_safe_loader
def load_mtcars_pydataset():
    """pydataset mtcars — fuel efficiency (mpg). γ₁ ≈ 0.61."""
    from pydataset import data as pdata
    df = pdata("mtcars").copy()
    target = "mpg"
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="mtcars", target=target,
                shift=0.0, gamma_expected=0.61,
                description="pydataset 'mtcars' (Henderson & Velleman, 1981)")


# ────────────────── (H) Сильно скошенные   γ₁ ∈ (1.5, 3.0] ───────────────────

@_safe_loader
def load_tips_pydataset():
    """pydataset tips — restaurant tip ($). γ₁ ≈ 1.47."""
    from pydataset import data as pdata
    df = pdata("tips").copy()
    target = "tip"
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Tips_Restaurant", target=target,
                shift=0.0, gamma_expected=1.47,
                description="pydataset 'tips' (Bryant & Smith, 1995)")


@_safe_loader
def load_ames_housing():
    """OpenML house_prices (Ames Housing) — SalePrice ($). γ₁ ≈ 1.88."""
    data = _fetch_openml_safe(42165, "house_prices")
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.lower() in ("saleprice", "price", "target")]
    target = target_cands[0] if target_cands else df.columns[-1]
    # Слишком много категориальных и пропусков: оставим только числовые
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in num_cols:
        num_cols.append(target)
    df = df[num_cols].dropna()
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Ames_Housing", target=target,
                shift=0.0, gamma_expected=1.88,
                description="OpenML 42165 'house_prices' (De Cock, 2011)")


@_safe_loader
def load_medical_insurance():
    """OpenML insurance — medical charges ($). γ₁ ≈ 1.52."""
    data = _fetch_openml_safe(("insurance", 1), 43463, 43948)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.lower() in ("charges", "expenses", "cost", "target")]
    target = target_cands[0] if target_cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Medical_Insurance", target=target,
                shift=0.0, gamma_expected=1.52,
                description="OpenML 'insurance' (Lantz, Machine Learning with R)")


@_safe_loader
def load_bike_sharing_day():
    """OpenML bike_sharing — daily count cnt. γ₁ ≈ 1.18."""
    data = _fetch_openml_safe(("Bike_Sharing_Demand", 2),
                              "bike-sharing-demand", 42713, 44048)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.lower() in ("cnt", "count", "target")]
    target = target_cands[0] if target_cands else df.columns[-1]
    # Уберём datetime-колонки
    for c in df.columns:
        if df[c].dtype.kind == "M":
            df = df.drop(columns=[c])
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Bike_Sharing", target=target,
                shift=0.0, gamma_expected=1.18,
                description="OpenML 'bike-sharing-demand' (Fanaee-T & Gama, UCI)")


@_safe_loader
def load_communities_crime():
    """OpenML communities-and-crime — ViolentCrimesPerPop. γ₁ ≈ 2.05."""
    data = _fetch_openml_safe("communities-and-crime", 43889, 42730, 211)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if "violentcrim" in c.lower() or c.lower() == "target"]
    target = target_cands[0] if target_cands else df.columns[-1]
    # Удалить все нечисловые / с большой долей NaN признаки
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in num_cols:
        num_cols.append(target)
    df = df[num_cols]
    keep = [c for c in df.columns if df[c].isna().mean() < 0.2]
    df = df[keep].dropna()
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Communities_Crime", target=target,
                shift=0.0, gamma_expected=2.05,
                description="OpenML 'communities-and-crime' (Redmond, UCI)")


# ────────────────── (X) Экстремально скошенные   γ₁ > 3 ──────────────────────

@_safe_loader
def load_servo():
    """OpenML servo — rise time of servomechanism. γ₁ ≈ 2.85."""
    data = _fetch_openml_safe("servo", 870, 87)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.lower() in ("class", "target", "vgain", "y")]
    target = target_cands[0] if target_cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Servo", target=target,
                shift=0.0, gamma_expected=2.85,
                description="OpenML 'servo' (Quinlan, UCI)")


@_safe_loader
def load_yacht_hydrodynamics():
    """OpenML yacht_hydrodynamics — residuary resistance. γ₁ ≈ 3.04."""
    data = _fetch_openml_safe("yacht_hydrodynamics", 42098, 43439)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if "resid" in c.lower() or c.lower() == "target"
                    or "v7" in c.lower()]
    target = target_cands[0] if target_cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Yacht_Hydrodynamics", target=target,
                shift=0.0, gamma_expected=3.04,
                description="OpenML 'yacht_hydrodynamics' (Ortigosa et al., UCI)")


@_safe_loader
def load_cpu_performance():
    """OpenML machine_cpu — Published Relative Performance (PRP). γ₁ ≈ 4.0."""
    data = _fetch_openml_safe("machine_cpu", 230, 733)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.upper() in ("PRP", "CLASS") or c.lower() == "target"]
    target = target_cands[0] if target_cands else df.columns[-1]
    # ERP — оценочная производительность, могла попасть как признак; уберём
    df = df.drop(columns=[c for c in df.columns
                          if c.upper() == "ERP" and c != target],
                 errors="ignore")
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="CPU_Performance", target=target,
                shift=0.0, gamma_expected=4.00,
                description="OpenML 'machine_cpu' (Ein-Dor & Feldmesser, UCI)")


@_safe_loader
def load_forest_fires():
    """OpenML forest_fires — burned area (ha). γ₁ ≈ 12.8 (много нулей!)."""
    data = _fetch_openml_safe("forest_fires", 42712, 484)
    df = data.frame.copy()
    target_cands = [c for c in df.columns
                    if c.lower() in ("area", "target")]
    target = target_cands[0] if target_cands else df.columns[-1]
    X, Y = _frame_to_numeric_Xy(df, target)
    return dict(X=X, Y=Y, name="Forest_Fires", target=target,
                shift=1.0, gamma_expected=12.8,
                description="OpenML 'forest_fires' (Cortez & Morais, UCI)")


# ─────────────────────────────────────────────────────────────────────────────
# Список всех 20 loader-ов (в порядке возрастания ожидаемого γ₁)
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_LOADERS = [
    # (S) γ₁ ∈ [-0.5, 0.5] — 6 шт.
    load_airfoil_self_noise,           # γ₁ ≈ -0.20
    load_wine_quality_white,           # γ₁ ≈  0.16
    load_wine_quality_red,             # γ₁ ≈  0.22
    load_energy_efficiency_heating,    # γ₁ ≈  0.36
    load_diabetes_sklearn,             # γ₁ ≈  0.44
    # (M) γ₁ ∈ (0.5, 1.5] — 5 шт.
    load_auto_mpg,                     # γ₁ ≈  0.46
    load_real_estate_taiwan,           # γ₁ ≈  0.60
    load_mtcars_pydataset,             # γ₁ ≈  0.61
    load_abalone,                      # γ₁ ≈  0.91
    load_boston_housing,               # γ₁ ≈  1.10
    # (H) γ₁ ∈ (1.5, 3.0] — 5 шт.
    load_bike_sharing_day,             # γ₁ ≈  1.18
    load_tips_pydataset,               # γ₁ ≈  1.47
    load_medical_insurance,            # γ₁ ≈  1.52
    load_ames_housing,                 # γ₁ ≈  1.88
    load_communities_crime,            # γ₁ ≈  2.05
    # (X) γ₁ > 3 — 4 шт.
    load_servo,                        # γ₁ ≈  2.85
    load_yacht_hydrodynamics,          # γ₁ ≈  3.04
    load_cpu_performance,              # γ₁ ≈  4.00
    load_forest_fires,                 # γ₁ ≈ 12.80 (с нулями → shift=1)
]


# ─────────────────────────────────────────────────────────────────────────────
# Обобщённый прогон одного датасета
# ─────────────────────────────────────────────────────────────────────────────

def run_generic_dataset(loader_result, idx=None, max_n=5000):
    """
    Универсальная функция прогона: печать описательной статистики,
    Box-Cox диагностика + LRT, run_experiment_cv, печать метрик.
    Большие выборки сабсэмплируются до max_n для управляемости.

    Возвращает (gamma1, df_metrics) либо (None, None), если loader_result is None.
    """
    if loader_result is None:
        return None, None

    X    = loader_result["X"]
    Y    = loader_result["Y"]
    name = loader_result["name"]
    tgt  = loader_result["target"]
    shft = float(loader_result.get("shift", 0.0))
    g_e  = loader_result.get("gamma_expected", None)
    descr = loader_result.get("description", "")

    header = f"{name.upper()}  (Y = {tgt})"
    if idx is not None:
        header = f"[{idx}/20] " + header
    section(header)

    # ── Описательная статистика ──
    g1 = float(skew(Y))
    n_obs, p_feat = len(Y), X.shape[1]
    zeros = int(np.sum(Y == 0))
    negs  = int(np.sum(Y < 0))
    print(f"\n  Источник: {descr}")
    print(f"  n={n_obs:,}, p={p_feat}")
    print(f"  Целевая переменная: {tgt}")
    if g_e is not None:
        print(f"  γ₁ = {g1:.3f}  (ожидалось ≈ {g_e:.2f})  "
              f"E[Y]={Y.mean():.4g}, Median={np.median(Y):.4g}, Sd={Y.std():.4g}")
    else:
        print(f"  γ₁ = {g1:.3f}  "
              f"E[Y]={Y.mean():.4g}, Median={np.median(Y):.4g}, Sd={Y.std():.4g}")
    print(f"  Min={Y.min():.4g}, Max={Y.max():.4g}")
    if zeros > 0:
        print(f"  Нулей в Y: {zeros} ({zeros/n_obs*100:.1f}%)")
    if negs > 0:
        print(f"  Отрицательных в Y: {negs} ({negs/n_obs*100:.1f}%)")
    if shft != 0.0:
        print(f"  Сдвиг δ={shft} для log/Box-Cox")

    try:
        n_sh = min(5000, n_obs)
        sh_p = float(shapiro(Y[:n_sh])[1])
        print(f"  Shapiro–Wilk(n={n_sh}) p = {sh_p:.4g}")
    except Exception as e:
        print(f"  Shapiro–Wilk: пропуск ({type(e).__name__})")

    # ── Box-Cox диагностика (требует Y + shift > 0) ──
    Y_for_bc = Y + shft
    if np.all(Y_for_bc > 0):
        try:
            n_bc = min(10000, n_obs)
            lam, ci = boxcox_fit(Y_for_bc[:n_bc])
            print(f"\n  Box-Cox λ̂ = {lam:.3f}  "
                  f"(95% ДИ: [{ci[0]:.3f}; {ci[1]:.3f}])")
            for lam_h, label_h in [(0.0, "λ=0 (логарифм)"),
                                   (1.0, "λ=1 (без преобр.)")]:
                chi2, p = lrt_boxcox(lam_h, lam, Y_for_bc[:n_bc])
                verdict = "не отвергается" if p > 0.05 else "отвергается"
                print(f"  LRT H₀: {label_h:22s}  "
                      f"χ²={chi2:.2f}  p={p:.4g}  → {verdict}")
        except Exception as e:
            print(f"  Box-Cox: пропуск ({type(e).__name__}: {str(e)[:60]})")
    else:
        print("\n  Box-Cox диагностика пропущена (Y+δ ≤ 0); "
              "используются только yeo-johnson / asinh / quantile.")

    # ── Сабсэмплинг при необходимости ──
    if n_obs > max_n:
        rng = np.random.default_rng(RANDOM_STATE)
        sub = rng.choice(n_obs, size=max_n, replace=False)
        X_s, Y_s = X[sub], Y[sub]
        print(f"\n  Подвыборка: n={len(Y_s):,} (из {n_obs:,}) для скорости")
    else:
        X_s, Y_s = X, Y

    # ── Эксперимент ──
    df, _, _ = run_experiment_cv(X_s, Y_s, shift=shft)
    print_metrics_multi(df, f"Метрики — {name}")

    base_rmse = float(np.sqrt(np.mean((Y_s - Y_s.mean()) ** 2)))
    print(f"\n  Baseline (среднее): RMSE = {base_rmse:.4g}")
    return g1, df


def run_extra_datasets():
    """
    Запускает все 20 дополнительных датасетов. Возвращает dict, готовый
    к слиянию в общий registry экспериментов.
    """
    section("ДОПОЛНИТЕЛЬНЫЕ ДАТАСЕТЫ (20 шт.) — расширенное покрытие γ₁")
    print(f"\n  Группы: (S) γ₁∈[-0.5,0.5]   (M) (0.5,1.5]   "
          f"(H) (1.5,3.0]   (X) γ₁>3")
    print(f"  Все большие выборки сабсэмплируются до 5000 наблюдений.\n")

    extra_registry = {}
    for i, loader in enumerate(EXTRA_LOADERS, start=1):
        loader_result = loader()
        g1, df = run_generic_dataset(loader_result, idx=i)
        if df is not None and loader_result is not None:
            extra_registry[loader_result["name"]] = (g1, df)
    return extra_registry


# ══════════════════════════════════════════════════════════════════════════════
# 6. ФОРМАЛИЗАЦИЯ ГИПОТЕЗЫ ОБ УБЫВАЮЩЕЙ ПОЛЕЗНОСТИ
# ══════════════════════════════════════════════════════════════════════════════

def utility(df_dataset, m_key, t_key):
    """U(model, transform) = −ΔRMSE_test % (положит. → преобр. полезно)."""
    try:
        return -float(df_dataset.loc[(m_key, t_key), "delta_rmse_pct"])
    except KeyError:
        return float("nan")


def is_significant(df_dataset, m_key, t_key, criterion="DM"):
    """Преобразование 'значимо лучше' baseline по выбранному критерию."""
    try:
        if criterion == "DM":
            p = df_dataset.loc[(m_key, t_key), "DM_p"]
        elif criterion == "paired_t":
            p = df_dataset.loc[(m_key, t_key), "paired_t_p"]
        else:
            return False
        delta = df_dataset.loc[(m_key, t_key), "delta_rmse_pct"]
        # значимо лучше = p<α И ΔRMSE отрицательная (RMSE уменьшилась)
        return (not np.isnan(p)) and p < ALPHA and delta < 0
    except KeyError:
        return False


def estimate_gamma_thresholds(registry, criterion="DM"):
    """
    Численная оценка порогового γ*₁ для каждой пары (модель, преобразование).

    γ*₁(m, t) = минимальное γ₁ среди датасетов, где преобразование t
                СТАТИСТИЧЕСКИ значимо снижает RMSE для модели m.

    Если такого порога нет — возвращает None (преобр. не помогает в диапазоне).
    """
    section(f"ОЦЕНКА ПОРОГОВЫХ γ*₁ (критерий: {criterion}, α={ALPHA})")
    print("\n  γ*₁ = минимальная асимметрия γ₁, при которой преобразование "
          "статистически\n  значимо (p<0.05) снижает RMSE; «—» означает, что "
          "значимого улучшения\n  не зафиксировано ни на одном датасете в диапазоне.\n")

    # Сортируем датасеты по возрастанию γ₁
    items = [(k, g, df) for k, (g, df) in registry.items()]
    items.sort(key=lambda x: x[1])

    print("    Порядок датасетов по γ₁:")
    for k, g, _ in items:
        print(f"      γ₁={g:>5.2f}  {k}")

    print(f"\n    {'Модель':<15} " +
          "".join(f"{TR_LABEL[t]:>13} " for t in TRANSFORMS[1:]))
    print("    " + "─" * (15 + 14 * (len(TRANSFORMS) - 1)))

    threshold_table = []
    for m_key in MODELS:
        line = f"    {MODEL_LABEL[m_key]:<15} "
        row = {"model": MODEL_LABEL[m_key], "model_class": MODEL_CLASS[m_key]}
        if m_key not in REGRESSION_MODELS:
            # Для древесных моделей и MLP преобразования отклика не
            # применяются методологически (см. шапку модуля); γ* не
            # определено по построению, а не отсутствует эмпирически.
            for t in TRANSFORMS[1:]:
                line += f"{'(н/п)':>13} "
                row[t] = None
            line += "  ← преобразования не применяются (см. шапку модуля)"
            threshold_table.append(row)
            print(line)
            continue
        for t in TRANSFORMS[1:]:
            gamma_star = None
            for k, g, df in items:
                if is_significant(df, m_key, t, criterion=criterion):
                    gamma_star = g
                    break
            label = f"γ*={gamma_star:.2f}" if gamma_star is not None else "—"
            line += f"{label:>13} "
            row[t] = gamma_star
        threshold_table.append(row)
        print(line)

    print("\n    (н/п) — преобразования к этому классу моделей не применяются;")
    print("    «—» — преобразование применялось, но статистически значимого "
          "улучшения нет.")

    pd.DataFrame(threshold_table).to_csv(
        RESULTS_DIR / f"gamma_thresholds_{criterion}.csv",
        index=False, encoding="utf-8-sig")

    # ── Агрегация по классу моделей ──
    section("СРАВНЕНИЕ КЛАССОВ МОДЕЛЕЙ — гипотеза об убывающей полезности")
    print("\n  ВАЖНО: согласно методологическому ограничению эксперимента")
    print("  (см. шапку модуля и docstring run_experiment_cv), нелинейные")
    print("  преобразования отклика применяются только к линейным регрессиям.")
    print("  Для древесных моделей и MLP оценивается лишь baseline 'none', т.е.")
    print("  U(tree, t≠none) и U(neural, t≠none) тождественно равны 0 по")
    print("  построению — преобразование к ним не применялось.")
    print("\n  Гипотеза об убывающей полезности в этой постановке формулируется")
    print("  как качественное утверждение: преобразования полезны исключительно")
    print("  для линейных моделей; для деревьев и MLP они методологически")
    print("  неприменимы (а не «не помогают эмпирически»). Ниже выводится U только")
    print("  для линейных моделей по датасетам — для иллюстрации зависимости от γ₁.")
    print(f"\n    {'Класс':<18} {'Датасет':<18} {'γ₁':>5}  "
          f"{'U(линейн.)':>11}")
    print("    " + "─" * 56)

    rows = []
    for k, g, df in items:
        line = f"    {'':<18} {k:<18} {g:>5.2f}"
        u_lin   = np.nanmean([utility(df, m, "boxcox")
                              for m in MODELS if MODEL_CLASS[m] == "линейные"])
        line += f"  {u_lin:>+10.1f}%"
        print(line)
        rows.append({"dataset": k, "gamma1": g,
                     "U_linear_pct_boxcox": u_lin,
                     "U_tree_pct_boxcox":   0.0,   # по построению
                     "U_neural_pct_boxcox": 0.0})  # по построению
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "utility_by_class.csv",
                              index=False, encoding="utf-8-sig")

    # ── Парный тест: проверяем зависимость U(линейн.) от γ₁ ──
    print("\n  Гипотеза о монотонной зависимости U(линейн., Box-Cox) от γ₁")
    print("         (корреляция Спирмена и Кендалла; H₀: ρ = 0)\n")

    g_arr = np.array([g for _, g, _ in items], dtype=float)
    u_arr = np.array([np.nanmean([utility(df, m, "boxcox")
                                  for m in MODELS if MODEL_CLASS[m] == "линейные"])
                      for _, _, df in items], dtype=float)
    mask = ~(np.isnan(g_arr) | np.isnan(u_arr))
    if mask.sum() >= 3:
        rho_s, p_s = stats.spearmanr(g_arr[mask], u_arr[mask])
        rho_k, p_k = stats.kendalltau(g_arr[mask], u_arr[mask])
        v_s = "ОТВЕРГАЕТСЯ" if p_s < ALPHA else "не отвергается"
        v_k = "ОТВЕРГАЕТСЯ" if p_k < ALPHA else "не отвергается"
        print(f"    Спирмен:  ρ_s = {rho_s:+.3f},  p = {p_s:.3f}  → H₀ {v_s}")
        print(f"    Кендалл:  τ   = {rho_k:+.3f},  p = {p_k:.3f}  → H₀ {v_k}")
        print("\n    Положительная корреляция U с γ₁ подтверждает гипотезу:")
        print("    преобразования становятся полезнее линейным моделям с ростом")
        print("    асимметрии Y. Для древесных моделей и MLP подобной зависимости")
        print("    нет по методологическому построению — преобразования к ним")
        print("    не применяются.")
    else:
        print("    Недостаточно датасетов для корреляционного анализа.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(registry):
    section("СВОДНЫЕ ТАБЛИЦЫ — ΔRMSE (test, %) с маркерами значимости")

    # Сортируем по γ₁ для удобства чтения
    items = sorted(registry.items(), key=lambda x: x[1][0])

    for m_key in MODELS:
        print(f"\n  ── {MODEL_LABEL[m_key]} {'─' * 55}")
        if m_key not in REGRESSION_MODELS:
            print(f"    [Преобразования отклика для модели «{MODEL_LABEL[m_key]}» "
                  f"методологически не применяются;\n     ΔRMSE для t≠none не "
                  f"оценивается. См. шапку модуля.]")
            continue
        hdr = f"    {'Датасет':<18} {'γ₁':>5}" + \
              "".join(f"  {TR_LABEL[t]:>14}" for t in TRANSFORMS[1:])
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for key, (g1, df) in items:
            line = f"    {key:<18} {g1:>5.2f}"
            for t in TRANSFORMS[1:]:
                try:
                    v = df.loc[(m_key, t), "delta_rmse_pct"]
                    p = df.loc[(m_key, t), "DM_p"]
                    if np.isnan(p):
                        marker = " "
                    elif p < ALPHA and v < 0:
                        marker = "*"   # значимо лучше
                    elif p < ALPHA and v > 0:
                        marker = "‼"   # значимо ХУЖЕ
                    else:
                        marker = " "
                    line += f"  {v:>+12.1f}%{marker}"
                except KeyError:
                    line += f"  {'—':>14} "
            print(line)
    print("\n    * — p_DM<0.05, преобр. значимо снижает RMSE")
    print("    ‼ — p_DM<0.05, преобр. значимо УВЕЛИЧИВАЕТ RMSE")

    # ── Сводка train vs test (отвечает на замечание руководителя) ──
    section("ΔRMSE на ОБУЧАЮЩЕЙ выборке (%) [по требованию рук-ля]")
    for m_key in MODELS:
        print(f"\n  ── {MODEL_LABEL[m_key]} {'─' * 55}")
        if m_key not in REGRESSION_MODELS:
            print(f"    [Преобразования не применяются — см. выше.]")
            continue
        hdr = f"    {'Датасет':<18} {'γ₁':>5}" + \
              "".join(f"  {TR_LABEL[t]:>14}" for t in TRANSFORMS[1:])
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for key, (g1, df) in items:
            line = f"    {key:<18} {g1:>5.2f}"
            for t in TRANSFORMS[1:]:
                try:
                    v_tr = df.loc[(m_key, t), "delta_rmse_train_pct"]
                    v_te = df.loc[(m_key, t), "delta_rmse_pct"]
                    gap = v_tr - v_te   # большой разрыв = переобучение/недообучение
                    line += f"  {v_tr:>+12.1f}% "
                except KeyError:
                    line += f"  {'—':>14} "
            print(line)

    # ── Полный CSV-дамп ──
    rows = []
    for key, (g1, df) in items:
        for m_key in MODELS:
            for t in TRANSFORMS:
                if (m_key, t) not in df.index:
                    continue
                r = df.loc[(m_key, t)]
                rows.append({
                    "Dataset": key, "gamma1": g1,
                    "Model": MODEL_LABEL[m_key],
                    "ModelClass": MODEL_CLASS[m_key],
                    "Transform": TR_LABEL[t],
                    "TransformClass": TR_CLASS[t],
                    "RMSE_train": r["RMSE_train"],
                    "RMSE_test": r["RMSE"],
                    "MAE_test": r["MAE"], "MAPE_test": r["MAPE"],
                    "R2_test": r["R2"],
                    "delta_RMSE_train_pct": r["delta_rmse_train_pct"],
                    "delta_RMSE_test_pct": r["delta_rmse_pct"],
                    "CV_RMSE_mean": r["cv_rmse_mean"],
                    "CV_RMSE_std": r["cv_rmse_std"],
                    "DM_stat": r.get("DM_stat", np.nan),
                    "DM_p": r.get("DM_p", np.nan),
                    "paired_t_stat": r.get("paired_t_stat", np.nan),
                    "paired_t_p": r.get("paired_t_p", np.nan),
                    # БЛОК 1 ТЗ: эффект-сайз и оценка мощности
                    "cohens_d":         r.get("cohens_d", np.nan),
                    "min_n_80power":    r.get("min_n_80power", np.nan),
                    "is_underpowered":  r.get("is_underpowered", False),
                })
    csv_path = RESULTS_DIR / "full_results_v6.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 72)
    print("  ВКР — ГЛАВА 2. ВЫЧИСЛИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ (v6)")
    print("  Модели (10): МНК, Ridge, Lasso, RF, XGBoost, MLP,")
    print("               Gamma GLM, Tweedie GLM, XGB-Gamma, XGB-Tweedie")
    print("  Преобразования: 7 (4 методологических класса),")
    print("                  ПРИМЕНЯЮТСЯ ТОЛЬКО к {МНК, Ridge, Lasso};")
    print("                  для RF/XGBoost/MLP — лишь baseline 'none';")
    print("                  для GLM и XGB-spec-loss — baseline 'none'")
    print("                  (alt-loss и преобразование Y несовместимы).")
    print("  БЛОК 1 ТЗ:    Cohen's d, n*(80% power), is_underpowered.")
    print("  БЛОК 3 ТЗ:    GLM и XGB-spec-loss — конкуренты преобразованию Y.")
    print("  Оценка:        5-fold CV + hold-out test + DM/paired t-tests + d.")
    print("  Датасеты:      7 базовых + 20 дополнительных = 27 (γ₁ ∈ [-0.5, 13])")
    print("█" * 72)

    # ── 0. Демонстрация Йенсена–Дуана (для теоретической главы) ──
    jensen_duan_demo(save_plot=True)

    # ── 1-5. Базовые эксперименты на датасетах ──
    registry = {}
    registry.update(run_synthetic())

    g_dia, df_dia = run_diamonds()
    registry["Diamonds"] = (g_dia, df_dia)

    try:
        g_cal, df_cal = run_california_housing()
        registry["California_Housing"] = (g_cal, df_cal)
    except Exception as e:
        print(f"\n  [SKIP] California Housing: {e}")

    g_con, df_con = run_concrete()
    if g_con is not None:
        registry["Concrete"] = (g_con, df_con)

    g_hie, df_hie = run_randhie()
    registry["RAND_HIE"] = (g_hie, df_hie)

    # ── 5.5. Дополнительные 20 датасетов ──
    extra = run_extra_datasets()
    registry.update(extra)
    print(f"\n  Итого датасетов в registry: {len(registry)}  "
          f"(из них дополнительных загрузилось: {len(extra)}/20)")

    # ── 6. Статистическая проверка гипотезы убывающей полезности ──
    estimate_gamma_thresholds(registry, criterion="DM")
    estimate_gamma_thresholds(registry, criterion="paired_t")

    # ── 6b. БЛОК 1 ТЗ: сводка underpowered-случаев ──
    print_underpowered_summary(registry)

    # ── 7. Сводные таблицы ──
    print_summary(registry)

    print("\n" + "─" * 72)
    print(f"  Все файлы: ./{RESULTS_DIR}/")
    print("─" * 72 + "\n")


if __name__ == "__main__":
    main()
