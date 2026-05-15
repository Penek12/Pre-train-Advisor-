"""
ВКР — Глава 3. БЛОК 4 ТЗ: Continuous advisor через мета-модель.

Назначение
──────────
В исходной KB рекомендации построены по 4 дискретным γ₁-бинам с жёсткими
границами 0.5 / 1.5 / 3.0. Это даёт два недостатка:
  • γ₁ = 0.49 и γ₁ = 0.51 получают совершенно разные рекомендации —
    артефакт дискретизации, не отражающий реальную плавность U(γ₁);
  • используется только одна характеристика данных (γ₁), хотя
    устойчивость λ̂_MLE (ширина 95%-ДИ), куртозис, доля нулей и размер
    выборки тоже информативны.

MetaModel решает обе проблемы:
  • обучает по одному регрессору на каждую пару (model_key, transform);
  • вход — 10-мерный вектор фич TargetDiagnostics;
  • выход — прогноз ΔRMSE_test% как непрерывная функция.

Используется LightGBM (если доступен) или GradientBoostingRegressor.
С 27 датасетами × 10 фич это работает (хотя и на грани — поэтому
консервативные гиперпараметры: max_depth=4, lr=0.05, n_estimators=200).

Качество мета-модели оценивается через LODO (см. chapter3_lodo): MAE
прогноза ΔRMSE на held-out датасетах. Если MAE мета-модели меньше MAE
бинного KB — континуальный advisor лучше дискретного.

Использование
─────────────
    python chapter3_meta_model.py                       # обучает и сохраняет
    python chapter3_meta_model.py --csv ...             # custom CSV

API:
    from chapter3_meta_model import MetaModel
    mm = MetaModel().fit_from_csv("results/full_results_v6.csv")
    pred = mm.predict(diag, model_key="linear", transform="boxcox")
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

try:
    from chapter3_advisor_v3 import (
        TargetDiagnostics, diagnose_target,
        LABEL_TO_TR, LABEL_TO_MODEL, ADVISOR_DIR,
    )
    from chapter2_experiments_v6 import (
        MODELS, MODEL_LABEL, TRANSFORMS, TR_LABEL, REGRESSION_MODELS,
        RANDOM_STATE,
    )
except ImportError as e:
    raise ImportError(
        "chapter3_meta_model.py требует chapter2_experiments_v6.py и "
        f"chapter3_advisor_v3.py в той же директории: {e}")

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Выбор бустингового регрессора: LightGBM приоритетно, GradientBoosting fallback
# ─────────────────────────────────────────────────────────────────────────────

def _make_meta_regressor():
    """Возвращает (regressor, name). Пытается LightGBM → fallback GBR."""
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_data_in_leaf=3,        # с 27 датасетами не больше
            num_leaves=8,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=1,
        )
        return m, "LightGBM"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        )
        return m, "sklearn.GradientBoostingRegressor"


# ═════════════════════════════════════════════════════════════════════════════
# MetaModel
# ═════════════════════════════════════════════════════════════════════════════

class MetaModel:
    """Continuous advisor: ансамбль бустинговых регрессоров,
    по одному на пару (model_key × transform).

    Целевая переменная — ΔRMSE_test% (как непрерывная величина).
    Фичи (10 шт.): gamma1, excess_kurt, log_n, p_features, zeros_pct, neg_pct,
                   boxcox_lambda, boxcox_ci_width, shapiro_p, mean_log_abs_Y.

    Хранит важности фич (feature_importances_) для интерпретируемости.
    """

    FEATURES: Tuple[str, ...] = (
        "gamma1", "excess_kurt", "log_n", "p_features",
        "zeros_pct", "neg_pct",
        "boxcox_lambda", "boxcox_ci_width",
        "shapiro_p", "mean_log_abs_Y",
    )

    def __init__(self):
        self._regressors: Dict[Tuple[str, str], object] = {}
        self._regressor_name: str = ""
        self._importance: Dict[Tuple[str, str], np.ndarray] = {}
        self._train_n: Dict[Tuple[str, str], int] = {}
        self._source: Optional[str] = None

    # ── Обучение из CSV ─────────────────────────────────────────────────────
    def fit_from_csv(self, csv_path: str,
                     target_col: str = "delta_RMSE_test_pct"
                     ) -> "MetaModel":
        """Обучает мета-модели по всем парам (model × transform) из CSV.

        Требует, чтобы CSV содержал:
          • колонки Dataset, gamma1, Model, Transform, {target_col};
          • желательно — n (n_train), p_features (число фич), shapiro_p,
            boxcox_lambda, boxcox_ci_lo, boxcox_ci_hi, zeros_pct, neg_pct.
            Если каких-то нет — они заполняются NaN, что для бустинга OK.

        Если колонок диагностики Y нет в CSV вообще — мета-модель будет
        опираться только на gamma1 (что хуже бинного KB, но всё же
        непрерывно).
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self._source = str(csv_path)

        # Маппинг моделей и преобразований
        df["Model_key"]     = df["Model"].map(LABEL_TO_MODEL)
        df["Transform_key"] = df["Transform"].map(LABEL_TO_TR)
        df = df.dropna(subset=["Model_key", "Transform_key"])

        # Baseline 'none' исключаем (ΔRMSE = 0 по построению)
        df = df[df["Transform_key"] != "none"].copy()

        # Подготовка фич: берём из CSV если есть, иначе NaN
        # gamma1 обязательно есть. Остальные — best-effort.
        feat_map = {
            "gamma1":          "gamma1",
            "excess_kurt":     "excess_kurt",       # может отсутствовать
            "log_n":           None,                # рассчитывается
            "p_features":      "p_features",        # может отсутствовать
            "zeros_pct":       "zeros_pct",
            "neg_pct":         "neg_pct",
            "boxcox_lambda":   "boxcox_lambda",
            "boxcox_ci_width": None,                # = ci_hi - ci_lo
            "shapiro_p":       "shapiro_p",
            "mean_log_abs_Y":  "mean_log_abs_Y",
        }

        # ── Если в CSV нет колонки 'n' — пробуем рассчитать из чего-то ──
        if "n" in df.columns:
            df["log_n"] = np.log(df["n"].astype(float).clip(lower=1))
        else:
            df["log_n"] = float("nan")

        # boxcox_ci_width
        if "boxcox_ci_lo" in df.columns and "boxcox_ci_hi" in df.columns:
            df["boxcox_ci_width"] = df["boxcox_ci_hi"].astype(float) - \
                                     df["boxcox_ci_lo"].astype(float)
        else:
            df["boxcox_ci_width"] = float("nan")

        # Гарантируем, что у нас есть все фичи (в т.ч. как NaN-колонки)
        for f in self.FEATURES:
            if f not in df.columns:
                df[f] = float("nan")

        # Обучаем по одному регрессору на каждую пару (mkey, tkey)
        pairs_trained = 0
        pairs_skipped = []
        for (mkey, tkey), sub in df.groupby(["Model_key", "Transform_key"]):
            X = sub[list(self.FEATURES)].astype(float).values
            y = sub[target_col].astype(float).values

            # Хотя бы 5 точек нужно для разумного бустинга. Меньше — пропускаем,
            # advisor для этой клетки будет использовать бинный KB.
            if len(y) < 5:
                pairs_skipped.append((mkey, tkey, len(y)))
                continue

            # Заполняем NaN в фичах медианой по столбцу (бустинги обычно сами
            # обрабатывают NaN, но LightGBM лучше с явным импутом; sklearn
            # GradientBoostingRegressor NaN не поддерживает).
            X = self._impute_features(X)

            reg, reg_name = _make_meta_regressor()
            self._regressor_name = reg_name
            try:
                reg.fit(X, y)
                self._regressors[(mkey, tkey)] = reg
                # feature_importances_ — общий API для LGBM и GBR
                imp = getattr(reg, "feature_importances_",
                              np.zeros(len(self.FEATURES)))
                if imp is not None and len(imp) == len(self.FEATURES):
                    # Нормализуем чтобы сумма = 1
                    s = imp.sum()
                    self._importance[(mkey, tkey)] = (imp / s
                                                      if s > 0 else imp)
                else:
                    self._importance[(mkey, tkey)] = np.zeros(len(self.FEATURES))
                self._train_n[(mkey, tkey)] = int(len(y))
                pairs_trained += 1
            except Exception as e:
                pairs_skipped.append((mkey, tkey, f"fit error: {type(e).__name__}"))
                continue

        if pairs_trained == 0:
            raise RuntimeError(
                "MetaModel: ни одной пары не обучено. Проверьте CSV "
                "(нужны минимум 5 датасетов на пару model × transform).")

        print(f"  MetaModel ({self._regressor_name}): обучено "
              f"{pairs_trained} пар (модель × преобр.); "
              f"пропущено {len(pairs_skipped)} (мало данных).")

        return self

    @staticmethod
    def _impute_features(X: np.ndarray) -> np.ndarray:
        """Импутация: NaN → медиана по столбцу (или 0, если все NaN)."""
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = ~np.isnan(col)
            if mask.any():
                X[~mask, j] = float(np.median(col[mask]))
            else:
                X[~mask, j] = 0.0
        return X

    # ── Прогноз ─────────────────────────────────────────────────────────────
    def predict(self, diag: TargetDiagnostics, model_key: str,
                transform: str, p_features: Optional[int] = None
                ) -> float:
        """Прогноз ΔRMSE_test% для пары (model_key, transform) на новых данных.

        Возвращает float (медиана прогноза). Если модели для этой пары нет —
        возвращает NaN (вызывающая сторона должна использовать бинный KB).
        """
        key = (model_key, transform)
        reg = self._regressors.get(key)
        if reg is None:
            return float("nan")

        # Собираем фичи в нужном порядке
        fv = diag.to_feature_vector()
        if p_features is not None:
            fv["p_features"] = float(p_features)
        x = np.array([[fv[f] for f in self.FEATURES]], dtype=float)
        x = self._impute_features(x)
        try:
            pred = float(reg.predict(x)[0])
        except Exception:
            pred = float("nan")
        return pred

    def predict_all(self, diag: TargetDiagnostics,
                    p_features: Optional[int] = None) -> pd.DataFrame:
        """Прогноз ΔRMSE для всех обученных пар (model × transform).

        Возвращает DataFrame с колонками Model_key, Transform_key,
        predicted_delta_pct, sorted по predicted_delta_pct (лучшее сверху).
        """
        rows = []
        for (mkey, tkey) in self._regressors.keys():
            pred = self.predict(diag, mkey, tkey, p_features=p_features)
            rows.append({"Model_key": mkey, "Transform_key": tkey,
                         "predicted_delta_pct": pred})
        out = pd.DataFrame(rows)
        return out.sort_values("predicted_delta_pct").reset_index(drop=True)

    # ── Важности признаков ──────────────────────────────────────────────────
    def feature_importance_table(self) -> pd.DataFrame:
        """Длинная таблица важностей: одна строка = одна (пара, фича)."""
        rows = []
        for (mkey, tkey), imp in self._importance.items():
            for f, v in zip(self.FEATURES, imp):
                rows.append({"Model_key": mkey, "Transform_key": tkey,
                             "feature": f, "importance": float(v)})
        return pd.DataFrame(rows)

    def feature_importance_aggregate(self) -> pd.DataFrame:
        """Усреднённые важности по всем парам — для итоговой таблицы в Главе 3."""
        df = self.feature_importance_table()
        if df.empty:
            return df
        return (df.groupby("feature")["importance"]
                  .agg(["mean", "std", "max"])
                  .reset_index()
                  .sort_values("mean", ascending=False))

    # ── Сохранение / загрузка ───────────────────────────────────────────────
    def save(self, model_path: Path, importance_csv: Optional[Path] = None
             ) -> Dict[str, Path]:
        """Сохраняет мета-модель в joblib + важности в CSV."""
        if joblib is None:
            raise RuntimeError("joblib не установлен; pip install joblib")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "regressors": self._regressors,
            "regressor_name": self._regressor_name,
            "importance": self._importance,
            "train_n": self._train_n,
            "source": self._source,
            "FEATURES": self.FEATURES,
        }, model_path)
        out = {"model": model_path}
        if importance_csv:
            self.feature_importance_table().to_csv(
                importance_csv, index=False, encoding="utf-8-sig")
            out["importance_csv"] = Path(importance_csv)
        return out

    @classmethod
    def load(cls, model_path: Path) -> "MetaModel":
        if joblib is None:
            raise RuntimeError("joblib не установлен; pip install joblib")
        data = joblib.load(model_path)
        mm = cls()
        mm._regressors = data["regressors"]
        mm._regressor_name = data.get("regressor_name", "unknown")
        mm._importance = data.get("importance", {})
        mm._train_n = data.get("train_n", {})
        mm._source = data.get("source")
        return mm

    def __repr__(self):
        return (f"MetaModel({self._regressor_name}, "
                f"{len(self._regressors)} pairs, source={self._source!r})")


# ═════════════════════════════════════════════════════════════════════════════
# LODO для мета-модели (сравнение с бинным KB)
# ═════════════════════════════════════════════════════════════════════════════

def lodo_metamodel(csv_path: str,
                   target_col: str = "delta_RMSE_test_pct"
                   ) -> Tuple[pd.DataFrame, dict]:
    """LODO-валидация мета-модели: для каждого датасета мета-модель
    обучается без него, и затем предсказывает ΔRMSE для пар (модель × преобр.)
    на этом датасете.

    Возвращает (df, summary) — тот же формат, что и chapter3_lodo.lodo_validate.
    Поскольку мета-модель НЕ имеет доступа к diag-фичам для held-out датасета
    (если CSV их не содержит), используются те же diag-колонки CSV, что и
    при обучении (gamma1, n, p_features, ...).

    Если diag-колонок в CSV нет, мета-модель будет опираться только на γ₁ —
    это всё ещё континуальный прогноз, но менее богатый.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df["Model_key"]     = df["Model"].map(LABEL_TO_MODEL)
    df["Transform_key"] = df["Transform"].map(LABEL_TO_TR)
    df = df.dropna(subset=["Model_key", "Transform_key"])
    df = df[df["Transform_key"] != "none"].copy()

    datasets = sorted(df["Dataset"].unique())
    records = []

    for i, hold_out in enumerate(datasets, start=1):
        print(f"    [{i}/{len(datasets)}] meta-model LODO, hold-out = {hold_out}")
        df_train = df[df["Dataset"] != hold_out]
        df_test  = df[df["Dataset"] == hold_out]
        if df_train.empty or df_test.empty:
            continue

        # Сохраняем во временный CSV и обучаем
        tmp = ADVISOR_DIR / f"_tmp_mm_excl_{hash(hold_out) & 0xFFFF}.csv"
        df_train.to_csv(tmp, index=False, encoding="utf-8-sig")
        try:
            mm = MetaModel().fit_from_csv(tmp, target_col=target_col)
        finally:
            try: tmp.unlink()
            except OSError: pass

        # Для каждой пары (mkey, tkey) на hold_out — предсказание из мета-модели
        for _, row in df_test.iterrows():
            mkey, tkey = row["Model_key"], row["Transform_key"]
            # Восстанавливаем фичи из CSV-строки (упрощённо: бёрем то,
            # что есть; недостающее — NaN)
            class _FakeDiag:
                pass
            fd = _FakeDiag()
            # Минимум — gamma1; остальное best-effort
            fd_dict = {
                "gamma1":          float(row.get("gamma1", float("nan"))),
                "excess_kurt":     float(row.get("excess_kurt", float("nan"))),
                "log_n":           float(np.log(max(float(row.get("n", 1)), 1)))
                                    if "n" in row else float("nan"),
                "p_features":      float(row.get("p_features", float("nan"))),
                "zeros_pct":       float(row.get("zeros_pct", float("nan"))),
                "neg_pct":         float(row.get("neg_pct", float("nan"))),
                "boxcox_lambda":   float(row.get("boxcox_lambda", float("nan"))),
                "boxcox_ci_width": (
                    float(row.get("boxcox_ci_hi", float("nan"))) -
                    float(row.get("boxcox_ci_lo", float("nan")))
                ) if "boxcox_ci_hi" in row else float("nan"),
                "shapiro_p":       float(row.get("shapiro_p", float("nan"))),
                "mean_log_abs_Y":  float(row.get("mean_log_abs_Y", float("nan"))),
            }
            fd.to_feature_vector = lambda d=fd_dict: d

            pred = mm.predict(fd, mkey, tkey)
            actual = float(row[target_col])
            records.append({
                "dataset":   hold_out,
                "gamma1":    float(row.get("gamma1", float("nan"))),
                "model":     MODEL_LABEL.get(mkey, mkey),
                "transform": TR_LABEL.get(tkey, tkey),
                "predicted_delta_pct": pred,
                "actual_delta_pct":    actual,
                "abs_error":   abs(actual - pred) if not np.isnan(pred)
                                                  else float("nan"),
                "sign_match":  int(np.sign(actual) == np.sign(pred)
                                   and abs(actual) > 0.5) if not np.isnan(pred)
                                                          else 0,
            })

    df_out = pd.DataFrame(records)
    if df_out.empty:
        return df_out, {}

    # Сводка
    valid = df_out.dropna(subset=["predicted_delta_pct"])
    summary = {
        "n_pairs":          int(len(valid)),
        "n_datasets":       int(df_out["dataset"].nunique()),
        "MAE_delta":        float(valid["abs_error"].mean()),
        "sign_match_rate":  float(valid["sign_match"].mean()),
    }
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(valid["predicted_delta_pct"],
                           valid["actual_delta_pct"])
        summary["spearman_correlation"] = float(rho)
        summary["spearman_p"] = float(p)
    except Exception:
        summary["spearman_correlation"] = float("nan")
        summary["spearman_p"] = float("nan")

    return df_out, summary


def compare_kb_vs_metamodel(csv_path: str) -> dict:
    """Сравнение бинного KB и мета-модели через LODO (БЛОК 4 ТЗ).

    Возвращает dict с двумя наборами метрик:
      {'kb_lodo': {...}, 'metamodel_lodo': {...},
       'delta_MAE': MAE_kb - MAE_metamodel}.
    Положительное delta_MAE ⇒ мета-модель лучше.
    """
    # Импортируем здесь чтобы избежать циклов
    from chapter3_lodo import lodo_validate
    df_kb, sum_kb = lodo_validate(csv_path, verbose=False)
    df_mm, sum_mm = lodo_metamodel(csv_path)

    return {
        "kb_lodo":        sum_kb.get("overall", {}),
        "metamodel_lodo": sum_mm,
        "delta_MAE": (sum_kb.get("overall", {}).get("MAE_delta", float("nan"))
                      - sum_mm.get("MAE_delta", float("nan"))),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MetaModel (continuous advisor) — БЛОК 4 ТЗ.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Путь к full_results_v6.csv.")
    parser.add_argument("--compare-lodo", action="store_true",
                        help="Сравнить мета-модель с бинным KB через LODO.")
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
    print("  ВКР — ГЛАВА 3, БЛОК 4: META-MODEL (CONTINUOUS ADVISOR)")
    print(f"  Источник: {csv_path}")
    print("█" * 72)

    mm = MetaModel().fit_from_csv(str(csv_path))
    print(f"\n  {mm}")

    print("\n  Топ-5 важных фич (усреднено по всем парам model × transform):")
    imp_agg = mm.feature_importance_aggregate()
    print(imp_agg.head(10).to_string(index=False))

    model_path = ADVISOR_DIR / "meta_model.joblib"
    imp_csv    = ADVISOR_DIR / "meta_model_feature_importance.csv"
    paths = mm.save(model_path, importance_csv=imp_csv)
    print(f"\n  Сохранено:")
    for k, p in paths.items():
        print(f"    {k}: {p}")

    if args.compare_lodo:
        print("\n  ── Сравнение бинного KB vs MetaModel через LODO ──")
        comp = compare_kb_vs_metamodel(str(csv_path))
        print(f"    KB:        MAE_delta = {comp['kb_lodo'].get('MAE_delta', float('nan')):.2f} pp")
        print(f"    MetaModel: MAE_delta = {comp['metamodel_lodo'].get('MAE_delta', float('nan')):.2f} pp")
        print(f"    Δ MAE (KB − MM)     = {comp['delta_MAE']:+.2f} pp "
              f"({'мета-модель лучше' if comp['delta_MAE'] > 0 else 'KB не хуже'})")
        with open(ADVISOR_DIR / "kb_vs_metamodel_lodo.json", "w",
                  encoding="utf-8") as f:
            def _san(o):
                if isinstance(o, dict):
                    return {k: _san(v) for k, v in o.items()}
                if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
                    return None
                return o
            json.dump(_san(comp), f, ensure_ascii=False, indent=2)

    print("\n  ✓ Готово.\n")


if __name__ == "__main__":
    main()
