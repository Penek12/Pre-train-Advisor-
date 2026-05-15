"""
Streamlit UI для advisor-а (Глава 3).

Запуск:
    streamlit run streamlit_app.py

Положите рядом с chapter3_advisor_v3.py и chapter2_experiments_v6.py.
Опционально — рядом results/full_results_v5.csv (KB из Главы 2).

Что умеет:
    1) Загрузка пользовательского CSV/XLSX датасета.
    2) Выбор колонки-отклика Y и фич X, выбор модели.
    3) Экспресс-диагностика отклика (γ₁, γ₂, Box-Cox λ, нули, Shapiro-Wilk).
    4) Top-K рекомендаций преобразований Y (с P10/P50/P90, prob_improvement).
    5) Сравнение трёх СТРАТЕГИЙ (transform_Y / GLM / specialized_loss).
    6) Опциональный empirical audit (5-fold CV) — фактическое ΔRMSE vs прогноз.
    7) Скачивание markdown-отчёта.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import probplot

# ── Импорт публичного API advisor-а ───────────────────────────────────────────
# Файл должен лежать рядом с chapter3_advisor_v3.py.
from chapter3_advisor_v3 import (
    KnowledgeBase,
    diagnose_target,
    recommend,
    recommend_strategy,
    audit,
    save_markdown_report,
    MODELS,
    MODEL_LABEL,
    MODEL_CLASS,
    REGRESSION_MODELS,
    TR_LABEL,
    GAMMA_BINS,
    ADVISOR_DIR,
    ALPHA,
)


# ═════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Advisor — рекомендации преобразований Y",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .stMetric { background: rgba(127,127,127,0.06); padding: .5rem .8rem;
                  border-radius: .5rem; }
      .rec-card { border: 1px solid rgba(127,127,127,0.25);
                  border-radius: .6rem; padding: 1rem 1.1rem; margin-bottom: .7rem;
                  background: rgba(127,127,127,0.04); }
      .rec-title { font-size: 1.05rem; font-weight: 600; margin-bottom: .25rem; }
      .rec-meta  { font-size: .88rem; opacity: .8; margin-bottom: .4rem; }
      .badge-ok  { background:#1f7a3a; color:white; padding:.12rem .55rem;
                   border-radius:1rem; font-size:.78rem; }
      .badge-no  { background:#a13a3a; color:white; padding:.12rem .55rem;
                   border-radius:1rem; font-size:.78rem; }
      .rationale { font-size: .9rem; opacity: .9; line-height: 1.45; }
      .warn { color:#b58200; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# КЭШИРОВАНИЕ ТЯЖЁЛЫХ ОПЕРАЦИЙ
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_kb_from_bytes(kb_bytes: bytes, fname: str) -> KnowledgeBase:
    """Грузит KB из загруженного пользователем CSV."""
    tmp = ADVISOR_DIR / f"_uploaded_{fname}"
    tmp.write_bytes(kb_bytes)
    return KnowledgeBase.from_csv(tmp)


@st.cache_data(show_spinner=False)
def _load_kb_default() -> KnowledgeBase:
    """Пробует загрузить KB из results/, иначе literature prior."""
    for p in [
        Path("results") / "full_results_v5.csv",
        Path("results") / "full_results_v4.csv",
    ]:
        if p.exists():
            try:
                return KnowledgeBase.from_csv(p)
            except Exception:
                pass
    return KnowledgeBase.from_defaults()


@st.cache_data(show_spinner=False)
def _read_user_file(file_bytes: bytes, fname: str) -> pd.DataFrame:
    """Читает CSV / Excel / TSV."""
    suf = Path(fname).suffix.lower()
    buf = io.BytesIO(file_bytes)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(buf)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(buf, sep="\t")
    # Авто-определение разделителя для .csv
    try:
        return pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf)


@st.cache_data(show_spinner="Считаю диагностику отклика…")
def _cached_diagnose(y_arr: np.ndarray) -> dict:
    """Кэш-обёртка над diagnose_target (возвращаем dict, чтобы хешировалось)."""
    d = diagnose_target(y_arr)
    return d.to_dict()


# ═════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ КОМПОНЕНТЫ UI
# ═════════════════════════════════════════════════════════════════════════════
def _y_distribution_chart(y: np.ndarray) -> go.Figure:
    """Гистограмма отклика + плотность."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y, nbinsx=60, name="Y",
        marker=dict(color="#4a90e2", line=dict(width=0)),
        opacity=0.85,
    ))
    fig.add_vline(x=float(np.median(y)), line=dict(color="#d97706", dash="dash"),
                  annotation_text=f"медиана = {np.median(y):.4g}",
                  annotation_position="top")
    fig.add_vline(x=float(np.mean(y)), line=dict(color="#16a34a", dash="dot"),
                  annotation_text=f"среднее = {np.mean(y):.4g}",
                  annotation_position="bottom")
    fig.update_layout(
        title="Распределение Y",
        xaxis_title="Y", yaxis_title="частота",
        height=320, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def _qq_chart(y: np.ndarray) -> go.Figure:
    """QQ-plot (Y vs нормальное распределение)."""
    osm, osr = probplot(y, dist="norm", fit=False)
    slope = float(np.std(y))
    intercept = float(np.mean(y))
    ref_x = np.array([osm[0], osm[-1]])
    ref_y = intercept + slope * ref_x
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                             marker=dict(size=4, color="#4a90e2"),
                             name="Y"))
    fig.add_trace(go.Scatter(x=ref_x, y=ref_y, mode="lines",
                             line=dict(color="#d97706", dash="dash"),
                             name="N(μ̂, σ̂²)"))
    fig.update_layout(
        title="QQ-plot (нормальное)",
        xaxis_title="теоретические квантили",
        yaxis_title="наблюдаемые квантили",
        height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
    )
    return fig


def _recs_interval_chart(recs) -> go.Figure:
    """Горизонтальные «эрорбары» P10/P50/P90 для top-K рекомендаций."""
    labels, p10s, p50s, p90s, colors = [], [], [], [], []
    for r in recs:
        labels.append(r.transform_label)
        p10s.append(r.delta_p10 if not np.isnan(r.delta_p10) else r.predicted_delta_pct)
        p50s.append(r.delta_p50 if not np.isnan(r.delta_p50) else r.predicted_delta_pct)
        p90s.append(r.delta_p90 if not np.isnan(r.delta_p90) else r.predicted_delta_pct)
        colors.append("#1f7a3a" if r.applicable and r.predicted_delta_pct < 0
                      else ("#a13a3a" if not r.applicable else "#888"))

    fig = go.Figure()
    for i, (lab, lo, mid, hi, col) in enumerate(
            zip(labels, p10s, p50s, p90s, colors)):
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lab, lab], mode="lines",
            line=dict(color=col, width=6), opacity=.45, showlegend=False,
            hovertemplate=f"[P10, P90] = [{lo:+.1f}%, {hi:+.1f}%]<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[mid], y=[lab], mode="markers",
            marker=dict(color=col, size=11, symbol="diamond"),
            showlegend=False,
            hovertemplate=f"медиана = {mid:+.1f}%<extra></extra>",
        ))
    fig.add_vline(x=0, line=dict(color="rgba(127,127,127,.5)", dash="dot"))
    fig.update_layout(
        title="Прогноз ΔRMSE_test% — медиана и интервал [P10, P90]",
        xaxis_title="ΔRMSE_test, %  (отрицательно = улучшение)",
        height=80 + 55 * len(labels),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _render_recommendation(r) -> None:
    """Карточка одной рекомендации (Recommendation)."""
    badge = ('<span class="badge-ok">применимо</span>' if r.applicable
             else '<span class="badge-no">не применимо</span>')
    p10 = r.delta_p10 if not np.isnan(r.delta_p10) else r.predicted_delta_pct
    p50 = r.delta_p50 if not np.isnan(r.delta_p50) else r.predicted_delta_pct
    p90 = r.delta_p90 if not np.isnan(r.delta_p90) else r.predicted_delta_pct
    ci_width = p90 - p10
    if ci_width < 10.0:
        ci_lbl = "узкий — надёжная рекомендация"
    elif ci_width < 25.0:
        ci_lbl = "умеренный"
    else:
        ci_lbl = "широкий — рискованная, большой разброс"

    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-title">#{r.rank} &nbsp; {r.transform_label}
              &nbsp; {badge}</div>
          <div class="rec-meta">класс: {r.transform_class} &nbsp;·&nbsp;
              n_evidence = {r.n_evidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("медиана ΔRMSE", f"{p50:+.1f}%")
    c2.metric("[P10, P90]", f"[{p10:+.1f}%, {p90:+.1f}%]", help=ci_lbl)
    c3.metric("P(улучшение)",
              f"{r.prob_improvement*100:.0f}%"
              if not np.isnan(r.prob_improvement) else "—")
    c4.metric("P(значимое улучшение)",
              f"{r.prob_significant*100:.0f}%"
              if not np.isnan(r.prob_significant) else "—")
    with st.expander("Обоснование и предупреждения"):
        st.markdown(f"<div class='rationale'>{r.rationale}</div>",
                    unsafe_allow_html=True)
        for w in r.warnings:
            st.markdown(f"<div class='warn'>⚠ {w}</div>",
                        unsafe_allow_html=True)


def _render_strategy(s) -> None:
    """Карточка одной стратегии (Strategy)."""
    badge = ('<span class="badge-ok">применимо</span>' if s.applicable
             else '<span class="badge-no">не применимо</span>')
    approach_lbl = {
        "transform_Y": "Преобразование Y + линейная модель",
        "glm": "GLM (Gamma / Tweedie, log-link)",
        "specialized_loss": "Спец. функция потерь (XGBoost Gamma/Tweedie)",
    }.get(s.approach, s.approach)

    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-title">#{s.rank} &nbsp; {s.model_label}
              + {s.transform_label} &nbsp; {badge}</div>
          <div class="rec-meta">подход: <b>{approach_lbl}</b>
              &nbsp;·&nbsp; n_evidence = {s.n_evidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("медиана ΔRMSE", f"{s.predicted_delta_pct:+.1f}%")
    if not np.isnan(s.delta_p10) and not np.isnan(s.delta_p90):
        c2.metric("[P10, P90]", f"[{s.delta_p10:+.1f}%, {s.delta_p90:+.1f}%]")
    else:
        c2.metric("[P10, P90]", "—")
    c3.metric("P(улучшение)",
              f"{s.prob_improvement*100:.0f}%"
              if not np.isnan(s.prob_improvement) else "—")
    c4.metric("P(значимое улучшение)",
              f"{s.prob_significant*100:.0f}%"
              if not np.isnan(s.prob_significant) else "—")
    with st.expander("Обоснование и предупреждения"):
        st.write(s.rationale)
        for w in s.warnings:
            st.markdown(f"<div class='warn'>⚠ {w}</div>",
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — настройка
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Параметры")
st.sidebar.markdown("**1. Данные**")

uploaded_file = st.sidebar.file_uploader(
    "Загрузите датасет (CSV / TSV / Excel)",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    help="Файл должен содержать колонку с откликом Y и хотя бы одну колонку-фичу.",
)

use_demo = st.sidebar.checkbox(
    "Использовать демо-датасет (Diamonds / синтетика)",
    value=(uploaded_file is None),
    help="Если файл не загружен — запустить advisor на демо-данных из главы 3.",
)

st.sidebar.markdown("**2. Knowledge Base**")
kb_file = st.sidebar.file_uploader(
    "Загрузите KB (full_results_v5.csv) — опционально",
    type=["csv"],
    help="Если не указано — будет искаться results/full_results_v5.csv "
         "или использован literature prior.",
)


# ═════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ СТРАНИЦА
# ═════════════════════════════════════════════════════════════════════════════
st.title("📈 Advisor: рекомендации преобразований отклика")
st.caption(
    "Pre-training рекомендации g(Y) для регрессионных задач — на основе "
    "27-датасетного бенчмарка (Глава 2) и теоретических корректировок "
    "(Box-Cox / Дуан / Йенсен)."
)

# ── Загрузка KB ───────────────────────────────────────────────────────────────
if kb_file is not None:
    kb = _load_kb_from_bytes(kb_file.getvalue(), kb_file.name)
    kb_source = f"загружен из {kb_file.name}"
else:
    kb = _load_kb_default()
    kb_source = kb.source
st.sidebar.success(f"KB: {kb_source}")

# ── Загрузка датасета ─────────────────────────────────────────────────────────
df: pd.DataFrame | None = None
data_label = ""

if uploaded_file is not None:
    try:
        df = _read_user_file(uploaded_file.getvalue(), uploaded_file.name)
        data_label = uploaded_file.name
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {type(e).__name__}: {e}")
        st.stop()
elif use_demo:
    from chapter3_advisor_v3 import _load_demo_data
    X_demo, y_demo, demo_label = _load_demo_data()
    df = pd.DataFrame(X_demo, columns=[f"x{i+1}" for i in range(X_demo.shape[1])])
    df["y"] = y_demo
    data_label = demo_label
else:
    st.info(
        "👈 Загрузите CSV/Excel-файл в боковой панели **или** включите "
        "галочку «Использовать демо-датасет», чтобы начать."
    )
    st.stop()

st.sidebar.markdown("**3. Колонки и модель**")

cols = list(df.columns)
# Эвристика выбора Y по умолчанию: колонка 'y' / 'price' / 'target', иначе последняя.
default_y = next((c for c in ["y", "price", "target", "Y"] if c in cols), cols[-1])
y_col = st.sidebar.selectbox(
    "Колонка отклика Y",
    cols, index=cols.index(default_y),
)
feature_cols = st.sidebar.multiselect(
    "Колонки фичей X",
    [c for c in cols if c != y_col],
    default=[c for c in cols if c != y_col],
)

model_key = st.sidebar.selectbox(
    "Модель",
    options=list(MODELS),
    format_func=lambda k: f"{MODEL_LABEL[k]}  ({MODEL_CLASS[k]})",
    index=list(MODELS).index("linear"),
)

top_k = st.sidebar.slider("Сколько рекомендаций показать (top-K)",
                          min_value=1, max_value=8, value=3)

st.sidebar.markdown("**4. Empirical audit (опционально)**")
do_audit = st.sidebar.checkbox(
    "Запустить эмпирическую проверку (5-fold CV)",
    value=False,
    help="Обучает модели для top-K преобразований и сравнивает фактическое "
         "ΔRMSE с прогнозом. Может занять 0.5–3 минуты в зависимости от n и модели.",
)


# ── Чтение y и применимость ───────────────────────────────────────────────────
try:
    y_series = pd.to_numeric(df[y_col], errors="coerce").dropna()
except Exception:
    st.error(f"Колонка '{y_col}' не приводится к числовому типу.")
    st.stop()

if len(y_series) < 30:
    st.error(f"В колонке '{y_col}' слишком мало числовых значений "
             f"(n={len(y_series)}). Нужно как минимум 30.")
    st.stop()

y_arr = y_series.values.astype(float)

# X — для empirical audit (только числовые фичи)
X_arr = None
if feature_cols:
    X_df = df.loc[y_series.index, feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    X_df = X_df.dropna(axis=1, how="all")  # выкинуть полностью пустые
    if X_df.shape[1] > 0:
        X_df = X_df.dropna()
        if len(X_df) >= 30:
            X_arr = X_df.values.astype(float)
            y_arr_for_audit = y_series.loc[X_df.index].values.astype(float)
        else:
            X_arr = None


# ═════════════════════════════════════════════════════════════════════════════
# ОСНОВНЫЕ ТАБЫ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"**Датасет:** {data_label} &nbsp;·&nbsp; **n** = {len(y_arr):,} "
    f"&nbsp;·&nbsp; **Y** = `{y_col}` &nbsp;·&nbsp; "
    f"**модель** = {MODEL_LABEL[model_key]}"
)

tab_diag, tab_recs, tab_strat, tab_audit, tab_report = st.tabs(
    ["📊 Диагностика Y",
     "🎯 Рекомендации преобразований",
     "🧩 Сравнение стратегий",
     "🔬 Empirical audit",
     "📥 Отчёт"]
)

# ── ТАБ 1: ДИАГНОСТИКА ────────────────────────────────────────────────────────
with tab_diag:
    diag_dict = _cached_diagnose(y_arr)
    diag = diagnose_target(y_arr)  # для передачи в render-функции (дёшево, кэш ниже)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("n", f"{diag.n:,}")
    col2.metric("γ₁ (асимметрия)", f"{diag.gamma1:+.3f}",
                help=diag.gamma_bin)
    col3.metric("γ₂ (экс-эксцесс)", f"{diag.excess_kurt:+.3f}",
                help="0 для нормального распределения")
    col4.metric("Shapiro-Wilk p", f"{diag.shapiro_p:.3g}",
                help="p < 0.05 ⇒ нормальность отвергается")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("E[Y]", f"{diag.mean:.4g}")
    col6.metric("Median", f"{diag.median:.4g}")
    col7.metric("Min", f"{diag.minv:.4g}")
    col8.metric("Max", f"{diag.maxv:.4g}")

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Нулей в Y", f"{diag.zeros_pct:.1f}%")
    col10.metric("Отрицательных в Y", f"{diag.neg_pct:.1f}%")
    col11.metric("Реком. сдвиг δ",
                 f"{diag.suggested_shift:.3g}" if diag.suggested_shift > 0
                 else "не нужен")
    if diag.boxcox_lambda is not None:
        col12.metric("Box-Cox λ̂", f"{diag.boxcox_lambda:.3f}",
                     help=f"95% ДИ: [{diag.boxcox_ci_lo:.3f}; "
                          f"{diag.boxcox_ci_hi:.3f}]")
    else:
        col12.metric("Box-Cox λ̂", "недоступно",
                     help="Y+δ имеет неположительные значения")

    # LRT (если доступно)
    if diag.boxcox_lambda is not None:
        st.markdown("**Likelihood-ratio тесты по Box-Cox:**")
        lrt_rows = []
        if diag.lrt_log_p is not None:
            lrt_rows.append({
                "H₀": "λ = 0 (логарифм)",
                "χ²": f"{diag.lrt_log_chi2:.3f}",
                "p-value": f"{diag.lrt_log_p:.4g}",
                "Вывод": "ОТВЕРГНУТА" if diag.lrt_log_p < ALPHA
                         else "не отвергнута",
            })
        if diag.lrt_none_p is not None:
            lrt_rows.append({
                "H₀": "λ = 1 (без преобразования)",
                "χ²": f"{diag.lrt_none_chi2:.3f}",
                "p-value": f"{diag.lrt_none_p:.4g}",
                "Вывод": "ОТВЕРГНУТА" if diag.lrt_none_p < ALPHA
                         else "не отвергнута",
            })
        if lrt_rows:
            st.dataframe(pd.DataFrame(lrt_rows), hide_index=True,
                         use_container_width=True)

    st.markdown(f"**Бин асимметрии:** {diag.gamma_bin}")

    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(_y_distribution_chart(y_arr), use_container_width=True)
    with c_right:
        st.plotly_chart(_qq_chart(y_arr), use_container_width=True)


# ── ТАБ 2: РЕКОМЕНДАЦИИ ─────────────────────────────────────────────────────
with tab_recs:
    st.subheader(f"Top-{top_k} преобразований для модели "
                 f"«{MODEL_LABEL[model_key]}»")

    if model_key not in REGRESSION_MODELS:
        st.warning(
            f"Модель «{MODEL_LABEL[model_key]}» — класса "
            f"«{MODEL_CLASS[model_key]}». Преобразования отклика к этому "
            f"классу не применяются методологически: теория Box-Cox / Дуана "
            f"требует линейности E[g(Y)|X] и гомоскедастичности остатков, "
            f"а Йенсеново смещение при инверсии у деревьев и MLP не "
            f"компенсируется поправкой Дуана. Для содержательного списка — "
            f"выберите Linear / Ridge / Lasso в боковой панели."
        )

    with st.spinner("Считаю рекомендации…"):
        _, recs = recommend(y_arr, model_key=model_key, kb=kb, top_k=top_k,
                            verbose=False)

    if model_key in REGRESSION_MODELS and len(recs) > 1:
        st.plotly_chart(_recs_interval_chart(recs), use_container_width=True)

    for r in recs:
        _render_recommendation(r)

    # Табличка для копирования
    with st.expander("Все рекомендации в табличном виде"):
        recs_df = pd.DataFrame([{
            "rank": r.rank,
            "transform": r.transform_label,
            "applicable": "✓" if r.applicable else "✗",
            "ΔRMSE_p50": f"{r.delta_p50:+.1f}%"
                if not np.isnan(r.delta_p50) else f"{r.predicted_delta_pct:+.1f}%",
            "[P10, P90]": (f"[{r.delta_p10:+.1f}%, {r.delta_p90:+.1f}%]"
                if not np.isnan(r.delta_p10) and not np.isnan(r.delta_p90)
                else "—"),
            "P(улучш.)": (f"{r.prob_improvement*100:.0f}%"
                if not np.isnan(r.prob_improvement) else "—"),
            "P(знач.)": (f"{r.prob_significant*100:.0f}%"
                if not np.isnan(r.prob_significant) else "—"),
            "n_evidence": r.n_evidence,
        } for r in recs])
        st.dataframe(recs_df, hide_index=True, use_container_width=True)


# ── ТАБ 3: СТРАТЕГИИ ────────────────────────────────────────────────────────
with tab_strat:
    st.subheader("Сравнение трёх подходов к асимметричному Y")
    st.caption(
        "• **transform_Y** — преобразование Y + линейная модель.  "
        "• **glm** — Gamma / Tweedie GLM с log-link.  "
        "• **specialized_loss** — XGBoost с reg:gamma / reg:tweedie."
    )
    with st.spinner("Сравниваю стратегии…"):
        _, strategies = recommend_strategy(y_arr, kb=kb, top_k=6,
                                           verbose=False)

    for s in strategies:
        _render_strategy(s)


# ── ТАБ 4: EMPIRICAL AUDIT ──────────────────────────────────────────────────
with tab_audit:
    st.subheader("Empirical audit — фактическое ΔRMSE на ваших данных")
    if not do_audit:
        st.info("Включите галочку «Запустить эмпирическую проверку» в боковой "
                "панели, чтобы запустить 5-fold CV для top-K преобразований.")
    elif X_arr is None:
        st.error("Для аудита нужны числовые фичи X (после очистки от NaN). "
                 "Проверьте выбор колонок фичей в боковой панели.")
    else:
        st.write(f"Запускаю 5-fold CV на n={len(y_arr_for_audit):,}, "
                 f"p={X_arr.shape[1]} фичах, для top-{top_k} преобразований…")
        with st.spinner("Обучаю модели… (это может занять до нескольких минут)"):
            try:
                _, recs_a, df_cv, actual = audit(
                    X_arr, y_arr_for_audit, model_key=model_key,
                    kb=kb, verbose=False,
                )
            except Exception as e:
                st.error(f"Ошибка во время аудита: {type(e).__name__}: {e}")
                df_cv = None

        if df_cv is not None:
            rows = []
            for r in recs_a:
                actual_d = actual.get(r.transform, float("nan"))
                diff = (actual_d - r.predicted_delta_pct
                        if not np.isnan(actual_d) else float("nan"))
                try:
                    dm_p = float(df_cv.loc[(model_key, r.transform), "DM_p"])
                    dm_p_s = f"{dm_p:.3f}"
                except Exception:
                    dm_p_s = "—"
                rows.append({
                    "Преобразование": r.transform_label,
                    "Прогноз ΔRMSE": f"{r.predicted_delta_pct:+.1f}%",
                    "Факт ΔRMSE": (f"{actual_d:+.1f}%"
                                   if not np.isnan(actual_d) else "—"),
                    "Δ (факт − прогноз)": (f"{diff:+.1f}pp"
                                          if not np.isnan(diff) else "—"),
                    "DM_p": dm_p_s,
                    "Применимо": "✓" if r.applicable else "✗",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

            # График: прогноз vs факт
            fig = go.Figure()
            for r in recs_a:
                actual_d = actual.get(r.transform, float("nan"))
                if np.isnan(actual_d):
                    continue
                fig.add_trace(go.Scatter(
                    x=[r.predicted_delta_pct], y=[actual_d],
                    mode="markers+text", text=[r.transform_label],
                    textposition="top center",
                    marker=dict(size=12,
                                color="#1f7a3a" if r.applicable else "#a13a3a"),
                    showlegend=False,
                ))
            lo, hi = -50, 30
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color="rgba(127,127,127,.4)", dash="dash"),
                name="y = x", showlegend=False,
            ))
            fig.update_layout(
                title="Прогноз vs факт (попадание на диагональ = идеальная калибровка)",
                xaxis_title="прогноз ΔRMSE, %",
                yaxis_title="факт ΔRMSE, %",
                height=420, margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── ТАБ 5: ОТЧЁТ ────────────────────────────────────────────────────────────
with tab_report:
    st.subheader("Markdown-отчёт")
    st.caption("Тот же отчёт, что генерирует `save_markdown_report` в advisor-е.")

    diag = diagnose_target(y_arr)
    _, recs_for_md = recommend(y_arr, model_key=model_key, kb=kb,
                               top_k=top_k, verbose=False)

    md_path = ADVISOR_DIR / f"report_{model_key}_streamlit.md"
    save_markdown_report(diag, recs_for_md, model_key, md_path)
    md_text = md_path.read_text(encoding="utf-8")

    st.download_button(
        "⬇️ Скачать report.md",
        data=md_text.encode("utf-8"),
        file_name=f"advisor_report_{model_key}.md",
        mime="text/markdown",
    )
    with st.expander("Предпросмотр отчёта"):
        st.markdown(md_text)


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "ВКР — Глава 3. Advisor v3 · "
    f"Knowledge base: {kb.source} · "
    "Pre-training рекомендации преобразований Y."
)
