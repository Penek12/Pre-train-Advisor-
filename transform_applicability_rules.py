"""
transform_applicability_rules.py — формальные методические правила
применимости преобразований отклика.

Источник: анализ файла full_results_v5.csv (26 датасетов × 10 моделей × 7
преобразований = 712 наблюдений) и связанных файлов:
  - gamma_thresholds_DM.csv, gamma_thresholds_paired_t.csv (пороги γ₁)
  - jensen_duan_*.csv (валидация поправки Дуана)

Четыре правила отвечают за четыре известных режима отказа, эмпирически
наблюдавшихся в бенчмарке:

  Rule-1 (model class)     — преобразования полезны только для линейных
                              моделей. Для деревьев, нейросетей и GLM —
                              блокировать целиком.
  Rule-2 (gamma threshold) — при |γ₁| < 0.45 эффект не превышает шума по
                              всем трём тестам (DM, paired-t, Cohen's d).
  Rule-3 (bounded range)   — Y, ограниченный сверху (например, [0, 1]),
                              ломает обратное log/Box-Cox преобразование
                              из-за нарушения симметрии остатков, нужной
                              поправке Дуана.  → Communities_Crime.
  Rule-4 (extreme gamma)   — при γ₁ ≥ 3 параметрические преобразования
                              (Box-Cox, Yeo-Johnson) численно нестабильны:
                              MLE по λ выходит в зону, где обратное
                              преобразование на тестовом наблюдении из
                              хвоста даёт NaN или 10¹¹⁺.
                              → CPU_Performance, RAND_HIE, Forest_Fires.

Каждое правило возвращает Verdict с явным reason-полем и confidence,
чтобы advisor мог отрисовать понятное обоснование в UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ ПРАВИЛ — числовые пороги, выведенные из бенчмарка
# ════════════════════════════════════════════════════════════════════════════

# Минимальный |γ₁|, начиная с которого преобразование статистически значимо
# улучшает RMSE хотя бы для одного линейного метода. Источник: пороги по DM
# и paired-t из gamma_thresholds_*.csv (медианное значение по непустым ячейкам).
GAMMA_MIN_FOR_EFFECT: float = 0.45

# Порог "экстремальной" асимметрии, за которым параметрические преобразования
# (Box-Cox, Yeo-Johnson) численно нестабильны. Эмпирически: γ₁ ≥ 3.86
# (CPU_Performance) уже даёт катастрофы; запас прочности — 3.0.
GAMMA_EXTREME: float = 3.0

# Признак "ограниченного" отклика: размах < 1 ИЛИ Y ⊂ [0, 1]. Включает все
# normalized indices, percentages, probabilities и т. п.
BOUNDED_RANGE_THRESHOLD: float = 1.0

# Классы моделей, для которых преобразования Y оправданы. Все остальные —
# Rule-1 блокирует целиком.
LINEAR_MODEL_CLASSES: frozenset = frozenset({"линейные", "linear"})

# Преобразования, которые могут провалиться на ограниченном Y (Rule-3),
# потому что их обратное преобразование (exp / power) разносит хвост.
UNBOUNDED_INVERSE_TRANSFORMS: frozenset = frozenset({"log", "boxcox"})

# Преобразования, нестабильные при экстремальной асимметрии (Rule-4).
# asinh и quantile численно стабильны и в этот список не входят.
PARAMETRIC_TRANSFORMS: frozenset = frozenset({"boxcox", "yeojohnson"})


# ════════════════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TargetStats:
    """Минимальная сводка отклика, нужная правилам.

    Создаётся либо вручную, либо из TargetDiagnostics advisor-а
    (см. TargetStats.from_diag).
    """
    gamma1: float                      # асимметрия
    min_y: float                       # min(Y)
    max_y: float                       # max(Y)
    zeros_pct: float = 0.0             # доля Y == 0, %
    n: int = 0                         # размер выборки

    @property
    def range(self) -> float:
        return self.max_y - self.min_y

    @property
    def is_bounded(self) -> bool:
        """Эвристика: размах < 1 ИЛИ Y явно в [0, 1]."""
        return (self.range < BOUNDED_RANGE_THRESHOLD
                or (self.min_y >= 0.0 and self.max_y <= 1.0))

    @classmethod
    def from_diag(cls, diag) -> "TargetStats":
        """Конструктор из TargetDiagnostics advisor-а (chapter3_advisor_v3)."""
        return cls(
            gamma1=float(diag.gamma1),
            min_y=float(diag.minv),
            max_y=float(diag.maxv),
            zeros_pct=float(diag.zeros_pct),
            n=int(diag.n),
        )

    @classmethod
    def from_array(cls, y: Iterable[float]) -> "TargetStats":
        """Конструктор из массива y (вычисляет статистики сам)."""
        from scipy.stats import skew
        y_arr = np.asarray(list(y), dtype=float)
        y_arr = y_arr[~np.isnan(y_arr)]
        return cls(
            gamma1=float(skew(y_arr, bias=False)),
            min_y=float(np.min(y_arr)),
            max_y=float(np.max(y_arr)),
            zeros_pct=float(100.0 * np.sum(y_arr == 0) / len(y_arr)),
            n=int(len(y_arr)),
        )


@dataclass
class Verdict:
    """Вердикт по одному преобразованию: применимо ли + почему."""
    transform: str                     # "log", "boxcox", "yeojohnson", ...
    allowed: bool                      # итоговое разрешение
    rule_triggered: Optional[str] = None  # "Rule-1" .. "Rule-4" или None
    reason: str = ""                   # человеческое объяснение
    severity: str = ""                 # "block" | "warn" | "ok"
    suggested_alternative: Optional[str] = None  # рекомендуемая замена

    def __repr__(self) -> str:
        if self.allowed:
            return f"<{self.transform}: OK>"
        return (f"<{self.transform}: BLOCKED by {self.rule_triggered} "
                f"→ {self.reason}>")


@dataclass
class RuleReport:
    """Полный отчёт по всем преобразованиям для одной задачи."""
    model_class: str
    stats: TargetStats
    verdicts: List[Verdict] = field(default_factory=list)
    global_block_reason: Optional[str] = None  # если Rule-1 или Rule-2

    @property
    def allowed_transforms(self) -> List[str]:
        return [v.transform for v in self.verdicts if v.allowed]

    @property
    def blocked_transforms(self) -> List[str]:
        return [v.transform for v in self.verdicts if not v.allowed]

    def explain(self) -> str:
        """Текстовое объяснение всех вердиктов (для UI / отчёта)."""
        lines = []
        if self.global_block_reason:
            lines.append(self.global_block_reason)
            lines.append("")
        for v in self.verdicts:
            tag = "✓" if v.allowed else "✗"
            line = f"  {tag} {v.transform}"
            if v.reason:
                line += f"  — {v.reason}"
            if v.suggested_alternative:
                line += f"  (см. {v.suggested_alternative})"
            lines.append(line)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# ЧЕТЫРЕ ПРАВИЛА — каждое как отдельная функция
# ════════════════════════════════════════════════════════════════════════════

def rule_1_model_class(model_class: str) -> Optional[Verdict]:
    """Rule-1: преобразования Y оправданы только для линейных моделей.

    Возвращает None, если правило не блокирует. Возвращает Verdict с
    severity='block', если модель не из класса линейных.

    Обоснование: в gamma_thresholds_DM.csv и gamma_thresholds_paired_t.csv
    все непустые ячейки находятся только в строках МНК / Ridge / Lasso.
    Для всех остальных классов (древесные, нейросетевые, glm,
    древесные-glm-loss) пороги — NaN, т. е. за весь бенчмарк ни одно
    преобразование не дало значимого улучшения.

    Теоретическое объяснение:
      * деревья восстанавливают нелинейность через разбиения,
        предварительное преобразование Y только сжимает диапазон;
      * GLM с естественной log-связью уже учитывают асимметрию через
        link-функцию, и дополнительное преобразование Y избыточно;
      * для нейросетей Йенсеново смещение при инверсии не компенсируется
        поправкой Дуана (Дуан выведен для линейных остатков).
    """
    if model_class in LINEAR_MODEL_CLASSES:
        return None
    return Verdict(
        transform="*",
        allowed=False,
        rule_triggered="Rule-1",
        severity="block",
        reason=(f"модель класса «{model_class}» не нуждается в "
                "преобразовании отклика: бенчмарк не выявил ни одного "
                "значимого улучшения для этого класса. Используйте "
                "сырой Y."),
    )


def rule_2_gamma_threshold(stats: TargetStats) -> Optional[Verdict]:
    """Rule-2: при |γ₁| < 0.45 преобразование не даёт эффекта.

    Возвращает None или Verdict с severity='block' и пояснением.

    Порог 0.45 = медиана непустых ячеек в gamma_thresholds_DM.csv и
    gamma_thresholds_paired_t.csv. Все три теста (DM, paired-t, Cohen's d)
    сходятся на этом значении.
    """
    if abs(stats.gamma1) >= GAMMA_MIN_FOR_EFFECT:
        return None
    return Verdict(
        transform="*",
        allowed=False,
        rule_triggered="Rule-2",
        severity="block",
        reason=(f"|γ₁| = {abs(stats.gamma1):.2f} < {GAMMA_MIN_FOR_EFFECT}. "
                "Бенчмарк из 27 датасетов не показал статистически "
                "значимых улучшений ниже этого порога ни по тесту "
                "Дибольда–Мариано, ни по paired t-test. Преобразование "
                "не оправдано."),
    )


def rule_3_bounded_range(transform: str, stats: TargetStats) -> Optional[Verdict]:
    """Rule-3: ограниченный Y запрещает log и Box-Cox.

    Возвращает None или Verdict с severity='block' и рекомендацией
    альтернативы.

    Механизм: при Y ∈ [0, 1] и значениях, близких к нулю, log(0.006) ≈ −5,
    log(0.95) ≈ −0.05 — два порядка разницы. После exp-инверсии один
    выброс с остатком ε ≈ +8 даёт прогноз с весом e⁸ ≈ 2980, RMSE
    взрывается. Поправка Дуана, требующая симметрии остатков, на
    ограниченном Y нарушается по построению.

    Эмпирический случай: Communities_Crime (γ₁ = 1.59, Y ∈ [0, 1]) —
    все шесть преобразований дали ΔRMSE > 10⁵%.
    """
    if not stats.is_bounded:
        return None
    if transform not in UNBOUNDED_INVERSE_TRANSFORMS:
        return None
    return Verdict(
        transform=transform,
        allowed=False,
        rule_triggered="Rule-3",
        severity="block",
        reason=(f"Y ограничен (диапазон = {stats.range:.3g}, "
                f"min = {stats.min_y:.3g}). Обратное преобразование {transform} "
                "разносит хвост при экспоненциальной инверсии (Йенсеново "
                "смещение, не компенсируемое поправкой Дуана)."),
        suggested_alternative="asinh или quantile",
    )


def rule_4_extreme_gamma(transform: str, stats: TargetStats) -> Optional[Verdict]:
    """Rule-4: при γ₁ ≥ 3 Box-Cox и Yeo-Johnson нестабильны.

    Возвращает None или Verdict с severity='block'.

    Механизм: MLE по λ при тяжёлых хвостах сходится к значениям,
    близким к 0, где обратное преобразование Y = (λ·ŷ + 1)^(1/λ) на
    тестовом наблюдении из хвоста даёт численное переполнение.

    Эмпирические случаи:
      * CPU_Performance (γ₁ = 3.86): Yeo-Johnson → ΔRMSE = +3.5×10⁸%
      * RAND_HIE (γ₁ = 4.82): Box-Cox → ΔRMSE = +8.5×10¹¹%
      * Forest_Fires (γ₁ = 12.81): Box-Cox → ΔRMSE = +4.8×10¹¹%
    """
    if stats.gamma1 < GAMMA_EXTREME:
        return None
    if transform not in PARAMETRIC_TRANSFORMS:
        return None
    return Verdict(
        transform=transform,
        allowed=False,
        rule_triggered="Rule-4",
        severity="block",
        reason=(f"γ₁ = {stats.gamma1:.2f} ≥ {GAMMA_EXTREME}. При такой "
                f"асимметрии MLE по λ нестабилен, обратное преобразование "
                f"{transform} на хвостовом наблюдении вылетает в ∞."),
        suggested_alternative=(
            "asinh или quantile, либо смена класса модели (GLM Gamma/Tweedie)"),
    )


# ════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ — ПРИМЕНЕНИЕ ВСЕХ ЧЕТЫРЁХ ПРАВИЛ
# ════════════════════════════════════════════════════════════════════════════

ALL_TRANSFORMS: List[str] = [
    "none", "log", "sqrt", "asinh", "boxcox", "yeojohnson", "quantile",
]


def check_applicability(
        stats: TargetStats,
        model_class: str,
        transforms: Optional[List[str]] = None,
        ) -> RuleReport:
    """Применить все четыре правила к каждому преобразованию.

    Параметры
    ─────────
    stats        : статистики отклика (TargetStats)
    model_class  : класс модели ("линейные", "древесные", "нейросетевые",
                   "glm", "древесные-glm-loss")
    transforms   : список преобразований для проверки.
                   По умолчанию все 7.

    Возвращает RuleReport со списком Verdict-ов и глобальным комментарием.

    Порядок правил важен: сначала глобальные блокировки (Rule-1, Rule-2),
    затем по-каждому-преобразованию (Rule-3, Rule-4). Это и есть «дерево
    решений» из главы 3.
    """
    if transforms is None:
        transforms = ALL_TRANSFORMS

    # ── Глобальные правила ───────────────────────────────────────────────
    global_block = rule_1_model_class(model_class)
    if global_block is None:
        global_block = rule_2_gamma_threshold(stats)

    report = RuleReport(
        model_class=model_class,
        stats=stats,
        global_block_reason=global_block.reason if global_block else None,
    )

    # ── Если глобальное правило сработало — блокируем все преобразования ─
    if global_block is not None:
        for t in transforms:
            if t == "none":
                report.verdicts.append(Verdict(
                    transform="none",
                    allowed=True,
                    severity="ok",
                    reason="используйте сырой Y",
                ))
            else:
                report.verdicts.append(Verdict(
                    transform=t,
                    allowed=False,
                    rule_triggered=global_block.rule_triggered,
                    severity=global_block.severity,
                    reason=global_block.reason,
                ))
        return report

    # ── Поэлементные правила (Rule-3, Rule-4) ────────────────────────────
    for t in transforms:
        # baseline всегда разрешён
        if t == "none":
            report.verdicts.append(Verdict(
                transform="none", allowed=True, severity="ok",
                reason="baseline без преобразования"))
            continue

        v3 = rule_3_bounded_range(t, stats)
        if v3 is not None:
            report.verdicts.append(v3)
            continue

        v4 = rule_4_extreme_gamma(t, stats)
        if v4 is not None:
            report.verdicts.append(v4)
            continue

        # Преобразование разрешено
        report.verdicts.append(Verdict(
            transform=t, allowed=True, severity="ok",
            reason=f"применимо по всем 4 правилам"))

    return report


# ════════════════════════════════════════════════════════════════════════════
# САМОПРОВЕРКА — четыре эмпирических случая из бенчмарка
# ════════════════════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Прогон правил на четырёх известных «дефектных» датасетах."""

    cases = [
        # (название, model_class, γ₁, min_y, max_y, ожидаемые блокировки)
        ("Communities_Crime", "линейные", 1.59, 0.0, 1.0,
         {"log", "boxcox"}),  # Rule-3: ограниченный Y
        ("CPU_Performance",   "линейные", 3.86, 6.0, 1150.0,
         {"boxcox", "yeojohnson"}),  # Rule-4: экстремальная асимметрия
        ("RAND_HIE",          "линейные", 4.82, 0.0, 39570.0,
         {"boxcox", "yeojohnson"}),  # Rule-4
        ("Forest_Fires",      "линейные", 12.81, 0.0, 1090.84,
         {"boxcox", "yeojohnson"}),  # Rule-4
        ("Auto_MPG",          "линейные", 0.46, 9.0, 46.6,
         set()),  # все разрешены
        ("WineQuality_Red",   "линейные", 0.22, 3.0, 8.0,
         {"log", "sqrt", "asinh", "boxcox", "yeojohnson", "quantile"}),
        # ↑ Rule-2: γ₁ < 0.45 ⇒ все блокируются
        ("California_RF",     "древесные", 0.98, 0.15, 5.0,
         {"log", "sqrt", "asinh", "boxcox", "yeojohnson", "quantile"}),
        # ↑ Rule-1: древесная модель ⇒ все блокируются
    ]

    print("=" * 72)
    print(f"{'CASE':<22} {'γ₁':>7}  {'class':<14} expected_blocks → actual")
    print("=" * 72)

    all_pass = True
    for name, mclass, g, lo, hi, expected_blocks in cases:
        stats = TargetStats(gamma1=g, min_y=lo, max_y=hi)
        report = check_applicability(stats, model_class=mclass)
        actual_blocks = set(report.blocked_transforms)

        ok = actual_blocks == expected_blocks
        mark = "✓" if ok else "✗ FAIL"
        all_pass &= ok

        print(f"{name:<22} {g:>+6.2f}  {mclass:<14} "
              f"{sorted(expected_blocks) or '∅'}")
        print(f"{' '*22} {'':>7}  {'':<14} "
              f"actual: {sorted(actual_blocks) or '∅'}  {mark}")
        print()

    print("=" * 72)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")


if __name__ == "__main__":
    _self_test()
