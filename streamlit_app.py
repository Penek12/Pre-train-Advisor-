"""
Streamlit UI для advisor-а. Редакционно-академический дизайн.

Запуск:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import probplot

from chapter3_advisor_v3 import (
    KnowledgeBase,
    diagnose_target,
    recommend,
    recommend_strategy,
    audit,
    empirical_full_audit,
    extract_rules_for_user,
    save_markdown_report,
    is_audit_available,
    MODELS,
    MODEL_LABEL,
    MODEL_CLASS,
    REGRESSION_MODELS,
    TR_LABEL,
    GAMMA_BINS,
    ADVISOR_DIR,
    ALPHA,
)

_AUDIT_AVAILABLE = is_audit_available()


# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГ + ПАЛИТРА
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advisor — преобразования отклика",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Палитра: тёплый кремовый фон, чернильный текст, ОДИН акцент (терракота),
# приглушённый шалфейный для «хороших» значений. Без неонов и без эмодзи.
INK       = "#1c1a17"   # текст
PAPER     = "#faf6ec"   # фон
RULE      = "#cfc6ad"   # хейрлайны
MUTED     = "#7a7361"   # вторичный текст
ACCENT    = "#8a3a25"   # терракота — единственный акцент
SAGE      = "#5c7a4f"   # «хорошо» (улучшение)
RUST      = "#9a4a2a"   # «плохо» (ухудшение)  — оттенок акцента, не красный
PAPER_DIM = "#f1ecdd"   # вторичный фон (таблицы, expanders)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;500&display=swap');

      /* Базовый сброс цветов под кремовый фон */
      html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
        background: {PAPER} !important;
        color: {INK} !important;
        font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;
      }}

      /* Скрываем стандартные виджеты Streamlit, чтобы не мешали редакционному виду */
      header[data-testid="stHeader"] {{ background: {PAPER}; }}
      div[data-testid="stToolbar"] {{ display: none; }}
      footer {{ display: none; }}
      #MainMenu {{ visibility: hidden; }}

      .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 1080px;
      }}

      /* Типографика */
      h1, h2, h3, h4, h5, h6 {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: {INK} !important;
        font-weight: 600 !important;
        letter-spacing: -0.005em;
      }}
      p, li, label, span, div {{
        color: {INK};
      }}

      /* Сайдбар */
      [data-testid="stSidebar"] {{
        background: {PAPER_DIM} !important;
        border-right: 1px solid {RULE};
      }}
      [data-testid="stSidebar"] * {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: {INK} !important;
      }}
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {{
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {MUTED} !important;
        margin-top: 1.5rem !important;
        margin-bottom: .5rem !important;
        padding-bottom: .25rem;
        border-bottom: 1px solid {RULE};
      }}

      /* Заголовок-«мастхед» */
      .masthead {{
        margin: .5rem 0 2rem 0;
        padding-bottom: 1.1rem;
        border-bottom: 2px solid {INK};
      }}
      .masthead-name {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: .35rem;
      }}
      .masthead-title {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        font-size: 2.0rem;
        line-height: 1.15;
        margin: 0;
        font-style: italic;
      }}
      .masthead-sub {{
        font-size: 1.0rem;
        color: {MUTED};
        margin-top: .35rem;
        font-style: italic;
        max-width: 65ch;
      }}

      /* Полоса метаданных датасета */
      .dataset-bar {{
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-size: .82rem;
        color: {MUTED};
        letter-spacing: 0.01em;
        margin: 0 0 2rem 0;
        padding: .55rem .9rem;
        border-left: 3px solid {ACCENT};
        background: {PAPER_DIM};
      }}
      .dataset-bar b {{ color: {INK}; font-weight: 500; }}

      /* Секционный заголовок: малые капс, хейрлайн снизу */
      .section {{
        margin-top: 2rem;
        margin-bottom: 1.1rem;
        padding-bottom: .35rem;
        border-bottom: 1px solid {RULE};
      }}
      .section-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: .68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: .25rem;
      }}
      .section-title {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        font-size: 1.35rem;
        margin: 0;
      }}

      /* Большие цифры как контент (вместо st.metric) */
      .stat {{
        margin: 0 0 1.4rem 0;
      }}
      .stat-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.3rem;
        line-height: 1;
        font-weight: 500;
        color: {INK};
        letter-spacing: -0.015em;
      }}
      .stat-num.accent {{ color: {ACCENT}; }}
      .stat-num.sage   {{ color: {SAGE}; }}
      .stat-num.rust   {{ color: {RUST}; }}
      .stat-label {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: .78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {MUTED};
        margin-top: .2rem;
      }}
      .stat-hint {{
        font-size: .82rem;
        color: {MUTED};
        font-style: italic;
        margin-top: .25rem;
      }}

      /* Прозовые врезки */
      .lede {{
        font-size: 1.04rem;
        line-height: 1.55;
        color: {INK};
        font-style: italic;
        max-width: 60ch;
        margin: 0 0 1.2rem 0;
      }}
      .note {{
        font-size: .92rem;
        line-height: 1.55;
        color: {MUTED};
        max-width: 65ch;
        margin: .5rem 0 1.3rem 0;
      }}

      /* Запись рекомендации — без карточек, через типографику */
      .rec {{
        padding: 1rem 0 1.2rem 0;
        border-bottom: 1px solid {RULE};
      }}
      .rec:last-child {{ border-bottom: none; }}
      .rec-num {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: .82rem;
        color: {ACCENT};
        letter-spacing: 0.06em;
        margin-right: .9rem;
        vertical-align: 1px;
      }}
      .rec-name {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        font-size: 1.2rem;
        color: {INK};
      }}
      .rec-name.dim {{ color: {MUTED}; text-decoration: line-through; }}
      .rec-meta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: .72rem;
        color: {MUTED};
        letter-spacing: 0.04em;
        margin-top: .15rem;
        margin-bottom: .8rem;
      }}
      .rec-figures {{
        display: flex;
        gap: 2.4rem;
        flex-wrap: wrap;
        margin: .4rem 0 .8rem 0;
      }}
      .rec-fig {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 500;
        color: {INK};
      }}
      .rec-fig small {{
        display: block;
        font-family: 'Source Serif 4', serif;
        font-size: .68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {MUTED};
        font-weight: 400;
        margin-top: .2rem;
      }}
      .rec-fig.accent {{ color: {ACCENT}; }}
      .rec-fig.sage   {{ color: {SAGE}; }}
      .rec-fig.rust   {{ color: {RUST}; }}
      .rec-rationale {{
        font-size: .95rem;
        line-height: 1.55;
        max-width: 70ch;
        color: {INK};
      }}

      /* Кнопки */
      .stButton > button {{
        background: {INK} !important;
        color: {PAPER} !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: .9rem !important;
        letter-spacing: 0.04em !important;
        padding: .55rem 1.4rem !important;
        font-weight: 500 !important;
      }}
      .stButton > button:hover {{
        background: {ACCENT} !important;
        color: {PAPER} !important;
      }}
      .stButton > button:disabled {{
        background: {PAPER_DIM} !important;
        color: {MUTED} !important;
      }}
      .stDownloadButton > button {{
        background: {PAPER} !important;
        color: {INK} !important;
        border: 1px solid {INK} !important;
        border-radius: 0 !important;
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: .88rem !important;
      }}
      .stDownloadButton > button:hover {{
        background: {INK} !important;
        color: {PAPER} !important;
      }}

      /* Вкладки */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: transparent;
        border-bottom: 2px solid {INK};
        margin-top: 0;
      }}
      .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: .9rem !important;
        font-weight: 500;
        color: {MUTED} !important;
        padding: .8rem 1.4rem !important;
        margin-right: .2rem;
        border-radius: 0 !important;
        letter-spacing: 0.02em;
      }}
      .stTabs [aria-selected="true"] {{
        color: {INK} !important;
        font-weight: 600 !important;
        border-bottom: 3px solid {ACCENT} !important;
        margin-bottom: -2px;
      }}

      /* Expander */
      [data-testid="stExpander"] {{
        border: 1px solid {RULE} !important;
        border-radius: 0 !important;
        background: {PAPER_DIM} !important;
        margin: 1rem 0 !important;
      }}
      [data-testid="stExpander"] summary {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-style: italic !important;
        color: {INK} !important;
      }}

      /* DataFrame: мягче */
      [data-testid="stDataFrame"] {{
        border: 1px solid {RULE};
        border-radius: 0;
      }}

      /* Селекты, чекбоксы — кремовый */
      [data-baseweb="select"] > div,
      [data-testid="stSelectbox"] > div > div,
      [data-testid="stMultiSelect"] > div > div {{
        background: {PAPER} !important;
        border-radius: 0 !important;
        border: 1px solid {RULE} !important;
      }}

      /* Сепаратор-флёрон между секциями */
      .fleuron {{
        text-align: center;
        color: {RULE};
        font-size: 1.5rem;
        margin: 2.2rem 0;
        font-family: 'Source Serif 4', serif;
        letter-spacing: 1rem;
      }}

      /* Колофон в подвале */
      .colophon {{
        margin-top: 4rem;
        padding: 1.4rem 0;
        border-top: 1px solid {RULE};
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: .8rem;
        font-style: italic;
        color: {MUTED};
        line-height: 1.55;
        max-width: 70ch;
      }}
      .colophon b {{ font-style: normal; color: {INK}; font-weight: 500; }}

      /* Алёрты */
      [data-testid="stAlert"] {{
        border-radius: 0 !important;
        border-left-width: 3px !important;
        font-family: 'Source Serif 4', Georgia, serif !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY-ТЕМА в тон палитре
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=PAPER,
    font=dict(family="Source Serif 4, Georgia, serif",
              color=INK, size=13),
    xaxis=dict(gridcolor=PAPER_DIM, linecolor=RULE, zerolinecolor=RULE,
               tickfont=dict(family="JetBrains Mono, monospace", size=11)),
    yaxis=dict(gridcolor=PAPER_DIM, linecolor=RULE, zerolinecolor=RULE,
               tickfont=dict(family="JetBrains Mono, monospace", size=11)),
    margin=dict(l=15, r=15, t=55, b=20),
)


# ─────────────────────────────────────────────────────────────────────────────
# КЭШИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_kb_from_bytes(kb_bytes: bytes, fname: str) -> KnowledgeBase:
    tmp = ADVISOR_DIR / f"_uploaded_{fname}"
    tmp.write_bytes(kb_bytes)
    return KnowledgeBase.from_csv(tmp)


@st.cache_data(show_spinner=False)
def _load_kb_default() -> KnowledgeBase:
    for p in [Path("results") / "full_results_v5.csv",
              Path("results") / "full_results_v4.csv"]:
        if p.exists():
            try:
                return KnowledgeBase.from_csv(p)
            except Exception:
                pass
    return KnowledgeBase.from_defaults()


@st.cache_data(show_spinner=False)
def _read_user_file(file_bytes: bytes, fname: str) -> pd.DataFrame:
    suf = Path(fname).suffix.lower()
    buf = io.BytesIO(file_bytes)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(buf)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(buf, sep="\t")
    try:
        return pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf)


# ─────────────────────────────────────────────────────────────────────────────
# КОМПОНЕНТЫ HTML (вместо st.metric)
# ─────────────────────────────────────────────────────────────────────────────
def section(eyebrow: str, title: str) -> None:
    """Заголовок секции: малые капс + хейрлайн."""
    st.markdown(
        f"""<div class="section">
          <div class="section-eyebrow">{eyebrow}</div>
          <div class="section-title">{title}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def stat(label: str, value: str, hint: str = "", tone: str = "") -> None:
    """Большая цифра + малые капс лейбл + курсивный hint."""
    tone_cls = f" {tone}" if tone in ("accent", "sage", "rust") else ""
    st.markdown(
        f"""<div class="stat">
          <div class="stat-num{tone_cls}">{value}</div>
          <div class="stat-label">{label}</div>
          {f'<div class="stat-hint">{hint}</div>' if hint else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def lede(text: str) -> None:
    """Прозовая курсивная врезка под заголовком."""
    st.markdown(f'<div class="lede">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    """Мелкая пояснительная заметка."""
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def fleuron() -> None:
    """Декоративный разделитель."""
    st.markdown('<div class="fleuron">· · ·</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ОПИСАНИЯ
# ─────────────────────────────────────────────────────────────────────────────
def _skew_prose(g: float, bin_label: str) -> str:
    """Человеческое описание асимметрии (вместо технического бина)."""
    if abs(g) < 0.5:
        return "распределение почти симметрично"
    direction = "правый" if g > 0 else "левый"
    if abs(g) < 1.0:
        return f"умеренный {direction} хвост"
    if abs(g) < 1.5:
        return f"выраженный {direction} хвост"
    if abs(g) < 3.0:
        return f"сильный {direction} хвост"
    return f"экстремальный {direction} хвост"


def _lambda_prose(lam: float) -> str:
    """Подсказка по Box-Cox λ̂."""
    if lam is None:
        return "λ̂ недоступна (Y содержит неположительные значения после сдвига)"
    if abs(lam) < 0.15:
        return "λ̂ ≈ 0 — оптимум близок к логарифму"
    if abs(lam - 0.5) < 0.15:
        return "λ̂ ≈ ½ — оптимум близок к √y"
    if abs(lam - 1.0) < 0.15:
        return "λ̂ ≈ 1 — преобразование почти не нужно"
    return f"λ̂ = {lam:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────────────
def _y_distribution_chart(y: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y, nbinsx=55,
        marker=dict(color=ACCENT, line=dict(width=0)),
        opacity=0.88,
    ))
    fig.add_vline(x=float(np.median(y)), line=dict(color=INK, dash="solid", width=1.2),
                  annotation_text=f"med {np.median(y):.3g}",
                  annotation_position="top",
                  annotation=dict(font=dict(family="JetBrains Mono", size=10)))
    fig.update_layout(**PLOT_LAYOUT,
                      title=dict(text="Распределение отклика",
                                 font=dict(size=14, family="Source Serif 4")),
                      height=300, showlegend=False,
                      xaxis_title="Y", yaxis_title=None)
    return fig


def _qq_chart(y: np.ndarray) -> go.Figure:
    osm, osr = probplot(y, dist="norm", fit=False)
    slope = float(np.std(y))
    intercept = float(np.mean(y))
    ref_x = np.array([osm[0], osm[-1]])
    ref_y = intercept + slope * ref_x
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref_x, y=ref_y, mode="lines",
                             line=dict(color=RULE, dash="dash", width=1.5),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                             marker=dict(size=4.5, color=ACCENT, opacity=0.6),
                             showlegend=False))
    fig.update_layout(**PLOT_LAYOUT,
                      title=dict(text="QQ-сравнение с нормальным",
                                 font=dict(size=14, family="Source Serif 4")),
                      height=300,
                      xaxis_title="теоретические квантили",
                      yaxis_title="наблюдаемые")
    return fig


def _recs_interval_chart(recs) -> go.Figure:
    """Тонкие линии P10-P90 + точка медианы. Без эрорбаров."""
    fig = go.Figure()
    for r in recs:
        lo = r.delta_p10 if not np.isnan(r.delta_p10) else r.predicted_delta_pct
        mid = r.delta_p50 if not np.isnan(r.delta_p50) else r.predicted_delta_pct
        hi = r.delta_p90 if not np.isnan(r.delta_p90) else r.predicted_delta_pct
        lab = r.transform_label
        col = SAGE if (r.applicable and mid < 0) else (RUST if not r.applicable else MUTED)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lab, lab], mode="lines",
            line=dict(color=col, width=2), opacity=.45, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[mid], y=[lab], mode="markers",
            marker=dict(color=col, size=10, symbol="line-ns",
                        line=dict(width=2, color=col)),
            showlegend=False,
        ))
    fig.add_vline(x=0, line=dict(color=INK, width=0.8, dash="dot"))
    fig.update_layout(**PLOT_LAYOUT,
                      title=dict(text="Прогноз ΔRMSE — медиана и интервал P10..P90",
                                 font=dict(size=14, family="Source Serif 4")),
                      xaxis_title="ΔRMSE, %  (отрицательно = улучшение)",
                      yaxis_title=None,
                      height=80 + 50 * len(recs))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### данные")

uploaded_file = st.sidebar.file_uploader(
    "CSV, TSV или Excel",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    label_visibility="visible",
)

use_demo = st.sidebar.checkbox(
    "взять демо-датасет",
    value=(uploaded_file is None),
)

st.sidebar.markdown("### база знаний")
kb_file = st.sidebar.file_uploader(
    "full_results_v5.csv (опц.)",
    type=["csv"],
)


# ─────────────────────────────────────────────────────────────────────────────
# МАСТХЕД
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="masthead">
      <div class="masthead-name">Advisor — Глава 3</div>
      <h1 class="masthead-title">Преобразования отклика<br>для регрессионных задач</h1>
      <div class="masthead-sub">
        Pre-training рекомендации g(Y) на основе бенчмарка из 27 датасетов,
        с теоретическими поправками Box–Cox, Дуана и Йенсена.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# ЗАГРУЗКА KB
# ─────────────────────────────────────────────────────────────────────────────
if kb_file is not None:
    kb = _load_kb_from_bytes(kb_file.getvalue(), kb_file.name)
    kb_source = f"загружена из {kb_file.name}"
else:
    kb = _load_kb_default()
    kb_source = kb.source

# ─────────────────────────────────────────────────────────────────────────────
# ЗАГРУЗКА ДАТАСЕТА
# ─────────────────────────────────────────────────────────────────────────────
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
    note(
        "Загрузите CSV-файл в боковой панели или включите «взять демо-датасет», "
        "чтобы начать. Инструмент работает с любой числовой колонкой-откликом "
        "и хотя бы одной числовой фичей."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ВЫБОР КОЛОНОК
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### колонки")
cols = list(df.columns)
default_y = next((c for c in ["y", "price", "target", "Y"] if c in cols), cols[-1])
y_col = st.sidebar.selectbox("отклик Y", cols, index=cols.index(default_y))
feature_cols = st.sidebar.multiselect(
    "фичи X", [c for c in cols if c != y_col],
    default=[c for c in cols if c != y_col],
)
model_key = st.sidebar.selectbox(
    "модель",
    options=list(MODELS),
    format_func=lambda k: f"{MODEL_LABEL[k]}",
    index=list(MODELS).index("linear"),
)
top_k = st.sidebar.slider("сколько рекомендаций", 1, 8, 3)

# Сноска про доступность audit
if _AUDIT_AVAILABLE:
    st.sidebar.markdown(
        f"<div style='font-size:.78rem;color:{MUTED};font-style:italic;"
        f"margin-top:1.5rem;padding-top:.8rem;border-top:1px solid {RULE}'>"
        "Эмпирическое сравнение стратегий доступно во вкладке "
        "<i>Сравнение стратегий</i>."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        f"<div style='font-size:.78rem;color:{MUTED};font-style:italic;"
        f"margin-top:1.5rem;padding-top:.8rem;border-top:1px solid {RULE}'>"
        "Облачный режим: обучение моделей недоступно. Доступны "
        "<i>Диагностика</i> и <i>Рекомендации</i>."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ВАЛИДАЦИЯ Y
# ─────────────────────────────────────────────────────────────────────────────
try:
    y_series = pd.to_numeric(df[y_col], errors="coerce").dropna()
except Exception:
    st.error(f"Колонка '{y_col}' не приводится к числовому типу.")
    st.stop()

if len(y_series) < 30:
    st.error(f"В колонке '{y_col}' слишком мало числовых значений "
             f"(n={len(y_series)}). Минимум — 30.")
    st.stop()

y_arr = y_series.values.astype(float)

X_arr = None
y_arr_for_audit = None
if feature_cols:
    X_df = df.loc[y_series.index, feature_cols].apply(
        pd.to_numeric, errors="coerce"
    ).dropna(axis=1, how="all")
    if X_df.shape[1] > 0:
        X_df = X_df.dropna()
        if len(X_df) >= 30:
            X_arr = X_df.values.astype(float)
            y_arr_for_audit = y_series.loc[X_df.index].values.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# ПОЛОСА МЕТАДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="dataset-bar">
      Работа с <b>{data_label}</b> &nbsp;·&nbsp;
      n = <b>{len(y_arr):,}</b> &nbsp;·&nbsp;
      отклик <b>{y_col}</b> &nbsp;·&nbsp;
      модель <b>{MODEL_LABEL[model_key]}</b> &nbsp;·&nbsp;
      KB: <b>{kb_source}</b>
    </div>""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# ВКЛАДКИ
# ─────────────────────────────────────────────────────────────────────────────
tab_diag, tab_recs, tab_strat, tab_report = st.tabs(
    ["Диагностика", "Рекомендации", "Сравнение стратегий", "Отчёт"]
)


# ═════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1. ДИАГНОСТИКА
# ═════════════════════════════════════════════════════════════════════════════
with tab_diag:
    diag = diagnose_target(y_arr)

    section("i", "Что говорит сам отклик")
    lede(
        f"Перед тем как принимать решение о преобразовании, посмотрим на "
        f"распределение Y без всякой модели. Из этих чисел уже видна "
        f"общая форма: {_skew_prose(diag.gamma1, diag.gamma_bin)}, "
        f"{'тяжёлые хвосты' if diag.excess_kurt > 1 else 'умеренные хвосты'}, "
        f"{_lambda_prose(diag.boxcox_lambda)}."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat("n", f"{diag.n:,}")
        hint = ''
    with c2:
        tone = "accent" if abs(diag.gamma1) > 1.0 else ""
        stat("γ₁ асимметрия", f"{diag.gamma1:+.2f}",
             hint=_skew_prose(diag.gamma1, diag.gamma_bin), tone=tone)
    with c3:
        tone = "accent" if diag.excess_kurt > 1.0 else ""
        stat("γ₂ эксцесс", f"{diag.excess_kurt:+.2f}",
             hint="избыток над нормальным распределением", tone=tone)
    with c4:
        if diag.boxcox_lambda is not None:
            stat("Box–Cox λ̂", f"{diag.boxcox_lambda:.2f}",
                 hint=f"95% ДИ: [{diag.boxcox_ci_lo:.2f}; {diag.boxcox_ci_hi:.2f}]",
                 tone="accent")
        else:
            stat("Box–Cox λ̂", "—", hint="Y содержит ≤0 после сдвига")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        stat("медиана", f"{diag.median:.3g}")
    with c6:
        stat("среднее", f"{diag.mean:.3g}",
             hint="разрыв с медианой = признак асимметрии")
    with c7:
        tone = "accent" if diag.zeros_pct > 0 else ""
        stat("нулей", f"{diag.zeros_pct:.0f}%",
             hint="блокирует log(y)" if diag.zeros_pct > 0 else "",
             tone=tone)
    with c8:
        tone = "rust" if diag.shapiro_p < 0.05 else ""
        stat("Shapiro p", f"{diag.shapiro_p:.3g}",
             hint="< 0.05 — нормальность отвергнута" if diag.shapiro_p < 0.05
                  else "не противоречит нормальности",
             tone=tone)

    # LRT (если есть)
    if diag.boxcox_lambda is not None and (diag.lrt_log_p or diag.lrt_none_p):
        section("ii", "Likelihood-ratio тесты по Box–Cox")
        note(
            "LRT отвечает на конкретный вопрос: «стоит ли отвергать частный "
            "случай λ» в пользу оптимума λ̂. Если p < 0.05, рассматриваемое "
            "преобразование статистически хуже оптимального; если нет — "
            "разумно использовать простой частный случай."
        )
        lrt_rows = []
        if diag.lrt_log_p is not None:
            verdict = "отвергается" if diag.lrt_log_p < ALPHA else "не отвергается"
            lrt_rows.append({"H₀": "λ = 0 (логарифм)",
                             "χ²": f"{diag.lrt_log_chi2:.2f}",
                             "p-value": f"{diag.lrt_log_p:.3g}",
                             "Вывод": verdict})
        if diag.lrt_none_p is not None:
            verdict = "отвергается" if diag.lrt_none_p < ALPHA else "не отвергается"
            lrt_rows.append({"H₀": "λ = 1 (без преобр.)",
                             "χ²": f"{diag.lrt_none_chi2:.2f}",
                             "p-value": f"{diag.lrt_none_p:.3g}",
                             "Вывод": verdict})
        st.dataframe(pd.DataFrame(lrt_rows), hide_index=True,
                     use_container_width=True)

    fleuron()

    section("iii", "Форма распределения")
    cL, cR = st.columns(2)
    with cL:
        st.plotly_chart(_y_distribution_chart(y_arr), use_container_width=True)
    with cR:
        st.plotly_chart(_qq_chart(y_arr), use_container_width=True)
    note(
        "Слева — гистограмма Y с отметкой медианы. Справа — QQ-сравнение с "
        "нормальным: если точки лежат на пунктире, распределение близко к "
        "нормальному, и преобразование вряд ли поможет. Отклонения по концам "
        "линии — признак хвостов."
    )


# ═════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2. РЕКОМЕНДАЦИИ
# ═════════════════════════════════════════════════════════════════════════════
with tab_recs:
    diag = diagnose_target(y_arr)
    section("i", f"Что советует база знаний для «{MODEL_LABEL[model_key]}»")
    lede(
        "Эта страница — чисто справочная. Ничего не обучается. Логика проста: "
        "по γ₁-бину отклика мы достаём из KB строки, соответствующие выбранной "
        "модели, и сортируем их по медиане ΔRMSE. Применимость проверяется "
        "отдельно (log запрещён при Y ≤ 0, и т. д.)."
    )

    if model_key not in REGRESSION_MODELS:
        st.warning(
            f"Модель «{MODEL_LABEL[model_key]}» — класса «{MODEL_CLASS[model_key]}». "
            "Преобразования отклика к этому классу не применяются по "
            "методологическим причинам: теория Box–Cox / Дуана выведена для "
            "линейной регрессии с гомоскедастичными остатками. Деревья сами "
            "улавливают нелинейность, а Йенсеново смещение при инверсии в "
            "нейросетях не компенсируется поправкой Дуана. Чтобы получить "
            "содержательный список — выберите Linear, Ridge или Lasso в "
            "боковой панели."
        )

    with st.spinner("Перебираю правила KB…"):
        _, recs = recommend(y_arr, model_key=model_key, kb=kb,
                            top_k=top_k, verbose=False)

    if model_key in REGRESSION_MODELS and len(recs) > 1:
        st.plotly_chart(_recs_interval_chart(recs), use_container_width=True)

    # Карточки рекомендаций — без боксов, через типографику
    for r in recs:
        applicable = r.applicable
        dim = "" if applicable else " dim"
        suffix = "" if applicable else ' <span style="color:'+RUST+';font-size:.78rem;font-style:italic">не применимо</span>'

        p10 = r.delta_p10 if not np.isnan(r.delta_p10) else r.predicted_delta_pct
        p50 = r.delta_p50 if not np.isnan(r.delta_p50) else r.predicted_delta_pct
        p90 = r.delta_p90 if not np.isnan(r.delta_p90) else r.predicted_delta_pct
        p50_tone = "sage" if (p50 < -2 and applicable) else (
                   "rust" if p50 > 2 else "")

        prob_imp = (f"{r.prob_improvement*100:.0f}%"
                    if not np.isnan(r.prob_improvement) else "—")
        prob_sig = (f"{r.prob_significant*100:.0f}%"
                    if not np.isnan(r.prob_significant) else "—")

        st.markdown(f"""
        <div class="rec">
          <div>
            <span class="rec-num">№{r.rank:02d}</span>
            <span class="rec-name{dim}">{r.transform_label}</span>
            {suffix}
          </div>
          <div class="rec-meta">{r.transform_class} · n={r.n_evidence}</div>
          <div class="rec-figures">
            <div class="rec-fig {p50_tone}">{p50:+.1f}%
              <small>медиана ΔRMSE</small></div>
            <div class="rec-fig">{p10:+.0f}…{p90:+.0f}
              <small>интервал P10..P90, %</small></div>
            <div class="rec-fig">{prob_imp}
              <small>P(улучшение)</small></div>
            <div class="rec-fig">{prob_sig}
              <small>P(значимое улучшение)</small></div>
          </div>
          <div class="rec-rationale">{r.rationale}</div>
        </div>
        """, unsafe_allow_html=True)

        if r.warnings:
            for w in r.warnings:
                st.markdown(
                    f'<div style="font-size:.85rem;font-style:italic;'
                    f'color:{RUST};margin:.3rem 0 .8rem 0;">— {w}</div>',
                    unsafe_allow_html=True)

    # Что лежит в KB — для прозрачности
    with st.expander(f"Сырые правила KB для бина «{diag.gamma_bin}»"):
        note(
            "Каждая строка — одно методическое правило: «для такой-то модели "
            "на датасетах с такой-то асимметрией данное преобразование в "
            "среднем даёт такой-то ΔRMSE». Эти правила выведены из 27 "
            "датасетов главы 2 и применены к вашему γ₁."
        )
        rules_all = extract_rules_for_user(diag, kb)
        if rules_all.empty:
            st.info("Для этого γ₁-бина в KB нет правил — "
                    "используется literature prior advisor-а.")
        else:
            rules_show = rules_all.copy()
            rules_show["·"] = rules_show["model_label"].apply(
                lambda m: "▸" if m == MODEL_LABEL[model_key] else "")
            cols_show = ["·", "model_label", "transform_label",
                         "median_delta_pct", "p10_delta_pct", "p90_delta_pct",
                         "sig_better_rate", "n_evidence"]
            cols_show = [c for c in cols_show if c in rules_show.columns]
            rules_show = rules_show[cols_show]
            rules_show.columns = ["·", "модель", "преобр.", "Δ медиана, %",
                                  "P10", "P90", "P(знач.)", "n"][:len(cols_show)]
            st.dataframe(rules_show, hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 3. СРАВНЕНИЕ СТРАТЕГИЙ
# ═════════════════════════════════════════════════════════════════════════════
with tab_strat:
    section("i", "Какая комбинация выигрывает по факту")
    lede(
        "В отличие от вкладки «Рекомендации», здесь обучаются все выбранные "
        "модели и применимые преобразования прямо на ваших данных. Победитель "
        "определяется не по KB-прогнозу, а по фактическому RMSE на hold-out."
    )

    if not _AUDIT_AVAILABLE:
        st.warning(
            "Эмпирическое сравнение в этом окружении недоступно — судя по "
            "всему, это облачный деплой без тяжёлых ML-зависимостей. Чтобы "
            "получить сравнение, склонируйте репозиторий, поставьте полный "
            "набор требований (requirements-full.txt) и запустите приложение "
            "локально."
        )
    elif X_arr is None:
        st.warning(
            "Для обучения нужны числовые фичи X. Сейчас после очистки от "
            f"пропусков их недостаточно — выбрано: {feature_cols or '∅'}."
        )
    else:
        st.markdown(
            f'<div class="note">Будет проведено '
            f"<b>5-fold CV + hold-out</b> для каждой модели × применимое "
            f"преобразование. Для линейных моделей — все 7 преобразований; "
            f"для остальных — только baseline (по методологическим причинам). "
            f"Данные: <b>n = {len(y_arr_for_audit):,}</b>, "
            f"<b>p = {X_arr.shape[1]}</b>.</div>",
            unsafe_allow_html=True,
        )

        default_models = ["linear", "ridge", "lasso", "rf", "xgb"]
        chosen_models = st.multiselect(
            "Модели для обучения",
            options=list(MODELS),
            default=[m for m in default_models if m in MODELS],
            format_func=lambda k: MODEL_LABEL[k],
        )

        c_run, c_eta = st.columns([1, 3])
        with c_run:
            run_btn = st.button("Запустить", type="primary",
                                disabled=not chosen_models)
        with c_eta:
            heavy = sum(1 for m in chosen_models
                        if m in ("mlp", "xgb", "rf", "xgb_gamma",
                                 "xgb_tweedie", "gamma_glm", "tweedie_glm"))
            light = len(chosen_models) - heavy
            eta_sec = int((1 * light + 8 * heavy)
                          * max(1.0, len(y_arr_for_audit) / 1000))
            if chosen_models:
                st.markdown(
                    f'<div style="font-family: \'JetBrains Mono\', monospace;'
                    f'font-size:.78rem;color:{MUTED};padding-top:.7rem;">'
                    f"≈ {eta_sec} сек · {len(chosen_models)} моделей · "
                    f"n = {len(y_arr_for_audit):,}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        cache_key = (data_label, tuple(chosen_models),
                     int(y_arr_for_audit.sum() * 1e6) % 10**9)

        if run_btn and chosen_models:
            progress_bar = st.progress(0.0, text="Готовлю обучение…")

            def _cb(i, total, label):
                frac = i / max(total, 1)
                txt = (f"Обучаю {label}… ({i}/{total})"
                       if label else f"Готово, {total} моделей.")
                progress_bar.progress(frac, text=txt)

            try:
                with st.spinner("Считаю…"):
                    diag_aud, df_ranked, winner = empirical_full_audit(
                        X_arr, y_arr_for_audit,
                        models=chosen_models,
                        progress_callback=_cb,
                        verbose=False,
                    )
                progress_bar.empty()
                st.session_state["audit_result"] = (cache_key, diag_aud,
                                                    df_ranked, winner)
            except Exception as e:
                progress_bar.empty()
                st.error(f"Ошибка: {type(e).__name__}: {e}")

        if ("audit_result" in st.session_state
                and st.session_state["audit_result"][0] == cache_key):
            _, diag_aud, df_ranked, winner = st.session_state["audit_result"]

            fleuron()
            section("ii", "Лучшая комбинация")

            if winner is None:
                st.error("Нет валидных результатов.")
            else:
                lede(
                    f"На ваших данных лучшим оказалось сочетание "
                    f"<b style='font-style:normal'>{winner['model_label']}</b> "
                    f"с преобразованием "
                    f"<b style='font-style:normal'>{winner['transform_label']}</b>. "
                    f"Все остальные комбинации даны ниже для сравнения."
                )

                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    stat("модель", winner["model_label"],
                         hint=f"класс: {winner['model_class']}")
                with w2:
                    stat("преобразование", winner["transform_label"])
                with w3:
                    stat("RMSE на test", f"{winner['RMSE']:.4g}",
                         tone="accent")
                with w4:
                    d = winner["delta_rmse_pct"]
                    if not np.isnan(d):
                        stat("ΔRMSE vs baseline", f"{d:+.1f}%",
                             tone=("sage" if d < 0 else "rust"))
                    else:
                        stat("ΔRMSE vs baseline", "—",
                             hint="baseline = none для той же модели")

                if winner["DM_p"] is not None:
                    dm = winner["DM_p"]
                    sig = "статистически значимое" if dm < ALPHA else "статистически не значимое"
                    st.markdown(
                        f'<div class="note">Diebold–Mariano p-value = '
                        f'<span style="font-family:\'JetBrains Mono\',monospace">'
                        f'{dm:.3g}</span> — {sig} различие с baseline ' 
                        f"(α = {ALPHA}).</div>",
                        unsafe_allow_html=True,
                    )

            section("iii", "Полное ранжирование")

            disp = df_ranked.reset_index().copy()
            show_cols = ["rank", "model_label", "transform_label",
                         "RMSE", "delta_rmse_pct", "DM_p"]
            show_cols = [c for c in show_cols if c in disp.columns]
            disp_show = disp[show_cols].copy()
            disp_show.columns = [
                {"rank": "№", "model_label": "модель",
                 "transform_label": "преобр.",
                 "RMSE": "RMSE",
                 "delta_rmse_pct": "ΔRMSE, %",
                 "DM_p": "DM p"}[c] for c in show_cols
            ]
            if "RMSE" in disp_show.columns:
                disp_show["RMSE"] = disp_show["RMSE"].map(
                    lambda x: f"{x:.4g}")
            if "ΔRMSE, %" in disp_show.columns:
                disp_show["ΔRMSE, %"] = disp_show["ΔRMSE, %"].map(
                    lambda x: f"{x:+.1f}" if not np.isnan(x) else "—")
            if "DM p" in disp_show.columns:
                disp_show["DM p"] = disp_show["DM p"].map(
                    lambda x: f"{x:.3g}" if not np.isnan(x) else "—")
            st.dataframe(disp_show, hide_index=True, use_container_width=True)

            # График: лучшая (transform) на модель
            section("iv", "Лучшее преобразование для каждой модели")
            best_per_model = df_ranked.reset_index().groupby("model").first()
            best_per_model = best_per_model.sort_values("RMSE")
            fig = go.Figure()
            colors = [SAGE if i == 0 else MUTED
                      for i in range(len(best_per_model))]
            fig.add_trace(go.Bar(
                x=[MODEL_LABEL[k] for k in best_per_model.index],
                y=best_per_model["RMSE"],
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"+ {TR_LABEL.get(t, t)}"
                      for t in best_per_model["transform"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color=MUTED),
                hovertemplate="<b>%{x}</b><br>RMSE = %{y:.4g}<extra></extra>",
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Победитель выделен шалфейным",
                           font=dict(size=13, family="Source Serif 4",
                                     color=MUTED)),
                xaxis_title=None, yaxis_title="RMSE на hold-out",
                height=380, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Сверка с KB
            with st.expander("Сверка: эмпирика vs прогноз KB"):
                note(
                    "Для каждой пары (модель, преобразование) сопоставлены "
                    "прогноз из KB (полученный по γ₁-бину) и фактический "
                    "результат на ваших данных. Большие отклонения указывают, "
                    "что обобщённые правила KB плохо переносятся на ваш "
                    "конкретный датасет."
                )
                rules_user = extract_rules_for_user(diag_aud, kb)
                if rules_user.empty:
                    st.info("KB не содержит правил для этого γ₁-бина.")
                else:
                    emp = df_ranked.reset_index()[
                        ["model_label", "transform_label", "delta_rmse_pct"]]
                    emp.columns = ["модель", "преобр.", "ΔRMSE факт, %"]
                    pred = rules_user[["model_label", "transform_label",
                                       "median_delta_pct"]].rename(
                        columns={"model_label": "модель",
                                 "transform_label": "преобр.",
                                 "median_delta_pct": "ΔRMSE KB, %"})
                    merged = pred.merge(emp, on=["модель", "преобр."],
                                        how="inner")
                    if not merged.empty:
                        merged["Δ, pp"] = (merged["ΔRMSE факт, %"]
                                           - merged["ΔRMSE KB, %"])
                        for c in ("ΔRMSE KB, %", "ΔRMSE факт, %", "Δ, pp"):
                            merged[c] = merged[c].map(
                                lambda x: f"{x:+.1f}"
                                if not np.isnan(x) else "—")
                        st.dataframe(merged, hide_index=True,
                                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 4. ОТЧЁТ
# ═════════════════════════════════════════════════════════════════════════════
with tab_report:
    section("i", "Markdown-отчёт")
    lede(
        "Тот же отчёт, что генерирует функция save_markdown_report из "
        "advisor-а — пригоден для вставки в дипломную работу или передачи "
        "коллегам без доступа к интерфейсу."
    )

    diag = diagnose_target(y_arr)
    _, recs_for_md = recommend(y_arr, model_key=model_key, kb=kb,
                               top_k=top_k, verbose=False)

    md_path = ADVISOR_DIR / f"report_{model_key}_streamlit.md"
    save_markdown_report(diag, recs_for_md, model_key, md_path)
    md_text = md_path.read_text(encoding="utf-8")

    st.download_button(
        "скачать report.md",
        data=md_text.encode("utf-8"),
        file_name=f"advisor_report_{model_key}.md",
        mime="text/markdown",
    )
    with st.expander("Предпросмотр"):
        st.markdown(md_text)


# ─────────────────────────────────────────────────────────────────────────────
# КОЛОФОН
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="colophon">
      <b>Advisor</b> — приложение к третьей главе ВКР об эмпирическом
      обосновании преобразований отклика для регрессионных задач.
      Набрано шрифтами Source Serif 4 и JetBrains Mono.
      База знаний: <i>{kb_source}</i>.
    </div>
    """,
    unsafe_allow_html=True,
)
