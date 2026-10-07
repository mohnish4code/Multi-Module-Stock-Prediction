# =========================================================
# MULTI-MODULE STOCK PREDICTION SYSTEM
# AI EQUITY TERMINAL  ·  Streamlit front-end
# =========================================================

import os
import sys
import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="AlphaStream · AI Equity Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from modules.data_collection import get_stock_data, get_quotes
from modules.technical_analysis import analyze_technical_indicators
from modules.inference import load_artifacts, load_metrics, predict_next_day
from modules.news_sentiment import analyze_company_sentiment
from modules.coordination_engine import coordinate_modules
from modules.confidence_engine import calculate_confidence
from modules.risk_engine import calculate_risk
from modules.recommendation_engine import get_final_recommendation
from config import COMPANIES, SEQUENCE_LENGTH, LIVE_DATA_PERIOD


# =========================================================
# THEME
# =========================================================

BG        = "#0a0e17"
SURFACE   = "#111725"
SURFACE_2 = "#161d2e"
BORDER    = "#1f2a3d"
TEXT      = "#dce3ef"
MUTED     = "#7d8ba3"
BULL      = "#16c784"
BEAR      = "#ea3943"
FLAT      = "#8b97ab"
ACCENT    = "#3b82f6"

st.markdown(
    f"""
    <style>
    #MainMenu, header, footer {{ visibility: hidden; }}
    .stApp {{ background: {BG}; color: {TEXT}; }}
    .block-container {{ padding: 0.6rem 1.4rem 3rem; max-width: 1550px; }}

    [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] * {{ color: {TEXT}; }}

    h1, h2, h3, h4 {{ color: #f2f5fa; letter-spacing: .2px; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}

    /* number / mono feel */
    .mono {{ font-variant-numeric: tabular-nums;
             font-family: "SF Mono","Roboto Mono",ui-monospace,Consolas,monospace; }}

    /* top bar */
    .topbar {{ display:flex; align-items:center; justify-content:space-between;
               padding:10px 16px; border:1px solid {BORDER}; border-radius:12px;
               background:linear-gradient(90deg,{SURFACE},{SURFACE_2}); margin-bottom:10px; }}
    .brand {{ font-weight:800; font-size:1.05rem; letter-spacing:1.5px; color:#f4f7fc; }}
    .brand small {{ color:{MUTED}; font-weight:600; letter-spacing:2px; margin-left:10px; font-size:.7rem; }}
    .clock {{ color:{MUTED}; font-size:.82rem; }}
    .dot {{ height:8px; width:8px; border-radius:50%; display:inline-block; margin-right:6px; }}

    /* ticker tape */
    .tape {{ display:flex; gap:22px; overflow-x:auto; padding:9px 14px; border:1px solid {BORDER};
             border-radius:12px; background:{SURFACE}; margin-bottom:14px; scrollbar-width:none; }}
    .tape::-webkit-scrollbar {{ display:none; }}
    .tape .cell {{ white-space:nowrap; font-size:.82rem; }}
    .tape .sym {{ color:{TEXT}; font-weight:700; margin-right:8px; }}
    .tape .px  {{ color:{MUTED}; margin-right:6px; }}

    /* cards */
    .card {{ border:1px solid {BORDER}; border-radius:14px; background:{SURFACE};
             padding:16px 18px; height:100%; }}
    .kpi-label {{ color:{MUTED}; font-size:.72rem; text-transform:uppercase; letter-spacing:1.2px; }}
    .kpi-value {{ font-size:1.55rem; font-weight:750; margin-top:4px; }}
    .kpi-sub   {{ color:{MUTED}; font-size:.78rem; margin-top:2px; }}

    .pill {{ display:inline-block; padding:4px 12px; border-radius:999px;
             font-weight:750; font-size:.8rem; letter-spacing:.5px; }}
    .sig-title {{ color:{MUTED}; font-size:.72rem; letter-spacing:1.4px; text-transform:uppercase; }}

    .rowline {{ display:flex; justify-content:space-between; padding:5px 0;
                border-bottom:1px dashed {BORDER}; font-size:.86rem; }}
    .rowline:last-child {{ border-bottom:none; }}
    .rowline .k {{ color:{MUTED}; }}

    /* news */
    .news {{ border:1px solid {BORDER}; border-left:3px solid {BORDER};
             border-radius:10px; padding:10px 14px; margin-bottom:8px; background:{SURFACE}; }}
    .news .h {{ font-weight:600; font-size:.92rem; color:{TEXT}; }}
    .news .m {{ color:{MUTED}; font-size:.75rem; margin-top:3px; }}

    /* sidebar radio -> instrument list */
    [data-testid="stSidebar"] .stRadio > div {{ gap:2px; }}
    [data-testid="stSidebar"] .stRadio label {{
        border:1px solid transparent; border-radius:9px; padding:7px 9px; width:100%;
        transition:.12s; }}
    [data-testid="stSidebar"] .stRadio label:hover {{ background:{SURFACE_2}; }}

    .stButton > button {{ width:100%; border-radius:10px; font-weight:750; letter-spacing:.6px;
        background:{ACCENT}; color:#fff; border:none; padding:.62rem; }}
    .stButton > button:hover {{ background:#2f6fe0; color:#fff; }}

    div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:12px; background:{SURFACE}; }}
    hr {{ border-color:{BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def colour_for(value):
    """Green/red/grey for a signed number OR a signal word."""
    if isinstance(value, (int, float)):
        return BULL if value > 0 else BEAR if value < 0 else FLAT
    return {"BULLISH": BULL, "BUY": BULL, "POSITIVE": BULL,
            "BEARISH": BEAR, "SELL": BEAR, "NEGATIVE": BEAR}.get(
        str(value).upper().strip(), FLAT)


def normalize_sentiment(label):
    label = str(label).upper().strip()
    return label if label in ("POSITIVE", "NEGATIVE", "NEUTRAL") else "NEUTRAL"


def ist_now():
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)


def market_state():
    now = ist_now()
    if now.weekday() >= 5:
        return "CLOSED", BEAR
    t = now.hour * 60 + now.minute
    if 555 <= t <= 930:          # 09:15 - 15:30 IST
        return "OPEN", BULL
    return "CLOSED", BEAR


@st.cache_data(ttl=300, show_spinner=False)
def cached_quotes(tickers):
    return get_quotes(tuple(tickers))


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(ticker):
    return get_stock_data(ticker=ticker, period=LIVE_DATA_PERIOD, interval="1d")


@st.cache_resource(show_spinner=False)
def cached_artifacts(model_folder):
    return load_artifacts(model_folder)


def kpi(col, label, value, sub="", value_colour=TEXT):
    col.markdown(
        f"""<div class="card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value mono" style="color:{value_colour}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def pill(text, colour):
    return (f'<span class="pill" style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}55">{text}</span>')


# =========================================================
# DATA FOR CHROME
# =========================================================

tickers = list(COMPANIES.keys())
quotes = cached_quotes(tuple(tickers))

mstate, mcolour = market_state()

st.markdown(
    f"""<div class="topbar">
        <div><span class="brand">◆ ALPHASTREAM</span><small>AI EQUITY TERMINAL</small></div>
        <div class="clock mono">
            <span class="dot" style="background:{mcolour}"></span>NSE {mstate}
            &nbsp;&nbsp;|&nbsp;&nbsp; {ist_now().strftime('%d %b %Y  %H:%M:%S')} IST
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ---- ticker tape ----
cells = []
for t in tickers:
    q = quotes.get(t)
    sym = t.replace(".NS", "")
    if not q:
        cells.append(f'<span class="cell"><span class="sym">{sym}</span>'
                     f'<span class="px">--</span></span>')
        continue
    c = colour_for(q["change_pct"])
    arrow = "▲" if q["change_pct"] > 0 else "▼" if q["change_pct"] < 0 else "•"
    cells.append(
        f'<span class="cell mono"><span class="sym">{sym}</span>'
        f'<span class="px">{q["price"]:,.2f}</span>'
        f'<span style="color:{c}">{arrow} {q["change_pct"]:+.2f}%</span></span>'
    )
st.markdown(f'<div class="tape">{"".join(cells)}</div>', unsafe_allow_html=True)


# =========================================================
# SIDEBAR  ·  INSTRUMENTS
# =========================================================

st.sidebar.markdown("### Instruments")

def _fmt(t):
    q = quotes.get(t)
    sym = t.replace(".NS", "")
    if q:
        c = "+" if q["change_pct"] >= 0 else ""
        return f"{sym:<11} {q['price']:>9,.0f}  {c}{q['change_pct']:.2f}%"
    return f"{sym:<11}      --"

selected = st.sidebar.radio(
    "instrument",
    tickers,
    format_func=_fmt,
    label_visibility="collapsed",
)

ticker = selected
company_name = COMPANIES[ticker]["name"]
model_folder = COMPANIES[ticker]["model_folder"]

st.sidebar.markdown("---")
run = st.sidebar.button("RUN AI ANALYSIS", width="stretch")

st.sidebar.markdown(
    f"""<div style="color:{MUTED};font-size:.76rem;line-height:1.7;margin-top:10px">
    <b style="color:{TEXT}">Pipeline</b><br>
    LSTM return forecast · Technical read · News sentiment<br>
    → coordination → confidence → risk → decision<br><br>
    <b style="color:{TEXT}">Data</b><br>
    Yahoo Finance EOD · Google News RSS (live)<br>
    Forecast horizon: next trading day
    </div>""",
    unsafe_allow_html=True,
)


# =========================================================
# INSTRUMENT HEADER
# =========================================================

q = quotes.get(ticker)
head_l, head_r = st.columns([3, 2])

with head_l:
    st.markdown(
        f"## {ticker.replace('.NS','')} "
        f"<span style='color:{MUTED};font-size:1rem;font-weight:500'>· {company_name}</span>",
        unsafe_allow_html=True,
    )

with head_r:
    if q:
        c = colour_for(q["change"])
        st.markdown(
            f"""<div style="text-align:right">
                <span class="mono" style="font-size:1.8rem;font-weight:750">{q['price']:,.2f}</span>
                <span class="mono" style="color:{c};font-size:1rem;margin-left:8px">
                    {q['change']:+,.2f} ({q['change_pct']:+.2f}%)</span>
                <div class="kpi-sub">last EOD close</div>
            </div>""",
            unsafe_allow_html=True,
        )


# =========================================================
# EMPTY STATE
# =========================================================

if not run:
    st.markdown(
        f"""<div class="card" style="margin-top:14px;text-align:center;padding:46px">
            <div style="font-size:2rem">◆</div>
            <div style="font-size:1.05rem;font-weight:650;margin-top:8px">
                {company_name} selected</div>
            <div class="kpi-sub" style="margin-top:6px">
                Press <b style="color:{TEXT}">RUN AI ANALYSIS</b> in the sidebar to generate the
                next-day forecast, technical read, live news sentiment and the coordinated call.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()


# =========================================================
# RUN THE PIPELINE
# =========================================================

try:
    prog = st.progress(0, text="Loading model artifacts...")

    artifacts = cached_artifacts(model_folder)
    model_metrics = load_metrics(model_folder)
    prog.progress(15, text="Fetching market data...")

    df = cached_history(ticker)
    if df is None or df.empty:
        raise ValueError("No market data received from the data provider.")
    prog.progress(35, text="Generating next-day forecast...")

    pred = predict_next_day(df, artifacts, sequence_length=SEQUENCE_LENGTH)
    df_features = pred["df_features"]

    current_price          = pred["current_price"]
    predicted_price        = pred["predicted_price"]
    price_change           = pred["predicted_change"]
    predicted_change_pct   = pred["predicted_change_percent"]
    prediction_direction   = pred["prediction_direction"]
    recommendation_signal  = pred["recommendation_signal"]
    prog.progress(55, text="Running technical analysis...")

    technical_result = analyze_technical_indicators(df_features)
    technical_signal = technical_result["technical_signal"]
    prog.progress(70, text="Fetching & scoring live news...")

    news_result = analyze_company_sentiment(company_name=company_name, ticker=ticker)
    news_sentiment = normalize_sentiment(news_result.get("sentiment_label", "NEUTRAL"))
    prog.progress(85, text="Coordinating modules...")

    coordination_result = coordinate_modules(
        prediction_direction, technical_signal, news_sentiment)
    agreement = coordination_result["agreement"]

    confidence_result = calculate_confidence(
        prediction_direction, technical_signal, news_sentiment, agreement)
    confidence_score = float(confidence_result["confidence_score"])
    confidence_level = confidence_result["confidence_level"]

    risk_result = calculate_risk(
        volatility=technical_result["volatility_level"],
        predicted_change_percent=predicted_change_pct,
        technical_signal=technical_signal,
        confidence_score=confidence_score,
        agreement=agreement,
    )
    risk_score = risk_result["risk_score"]
    risk_level = risk_result["risk_level"]

    final_result = get_final_recommendation(recommendation_signal, news_sentiment)
    recommendation = final_result.get("recommendation", recommendation_signal)

    prog.progress(100, text="Done")
    prog.empty()

except Exception as error:  # noqa: BLE001
    st.error("The analysis could not be completed.")
    st.exception(error)
    st.stop()


# =========================================================
# SIGNAL BADGE  +  KPI ROW
# =========================================================

rec_colour = colour_for(recommendation)
st.markdown(
    f"<div style='margin:6px 0 14px'>AI SIGNAL &nbsp; {pill(recommendation, rec_colour)} "
    f"<span class='kpi-sub'>&nbsp; coordinated: {coordination_result['coordinated_signal']} "
    f"· {agreement.lower()} agreement</span></div>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Predicted Close",
    f"₹{predicted_price:,.2f}",
    f"model target · next session", TEXT)
kpi(k2, "Expected Move",
    f"{predicted_change_pct:+.2f}%",
    f"₹{price_change:+,.2f} vs last close", colour_for(price_change))
if model_metrics and model_metrics.get("conviction_accuracy") is not None:
    kpi(k3, "Model Conviction Acc.",
        f"{model_metrics['conviction_accuracy']:.1f}%",
        f"top {model_metrics.get('conviction_coverage', 33):.0f}% of back-test days",
        BULL if model_metrics["conviction_accuracy"] >= 52 else FLAT)
else:
    kpi(k3, "Model Conviction Acc.", "N/A", "run training/retrain_all.py")
kpi(k4, "Risk Level", risk_level,
    f"score {risk_score}/10",
    BEAR if risk_level == "HIGH" else FLAT if risk_level == "MEDIUM" else BULL)


# =========================================================
# PRICE CHART  (candles + MA + Bollinger + forecast)
# =========================================================

st.markdown("#### Price action & AI forecast")

cd = df_features.tail(90).copy()
nxt = cd.index[-1] + pd.tseries.offsets.BDay(1)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
    vertical_spacing=0.04,
)

fig.add_trace(go.Candlestick(
    x=cd.index, open=cd["Open"], high=cd["High"], low=cd["Low"], close=cd["Close"],
    name="OHLC", increasing_line_color=BULL, decreasing_line_color=BEAR,
    increasing_fillcolor=BULL, decreasing_fillcolor=BEAR, line=dict(width=1),
), row=1, col=1)

if "BB_Upper" in cd:
    fig.add_trace(go.Scatter(x=cd.index, y=cd["BB_Upper"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cd.index, y=cd["BB_Lower"], line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.08)",
                             showlegend=False, hoverinfo="skip"), row=1, col=1)

for col_name, colr, dash in (("SMA_20", "#f0b90b", "solid"), ("EMA_20", "#8b5cf6", "solid")):
    if col_name in cd:
        fig.add_trace(go.Scatter(x=cd.index, y=cd[col_name], name=col_name,
                                 line=dict(color=colr, width=1.3, dash=dash)), row=1, col=1)

# forecast segment
fig.add_trace(go.Scatter(
    x=[cd.index[-1], nxt], y=[float(cd["Close"].iloc[-1]), predicted_price],
    name="AI forecast", mode="lines+markers",
    line=dict(color=rec_colour, width=2, dash="dot"),
    marker=dict(size=9, color=rec_colour, symbol="diamond"),
), row=1, col=1)
fig.add_annotation(x=nxt, y=predicted_price, text=f"  ₹{predicted_price:,.0f}",
                   showarrow=False, xanchor="left", font=dict(color=rec_colour, size=12), row=1, col=1)

vol_colours = [BULL if c >= o else BEAR for o, c in zip(cd["Open"], cd["Close"])]
fig.add_trace(go.Bar(x=cd.index, y=cd["Volume"], marker_color=vol_colours,
                     opacity=0.45, name="Volume"), row=2, col=1)

fig.update_layout(
    height=520, template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
    hovermode="x unified",
)
fig.update_xaxes(gridcolor=BORDER, showspikes=True, spikecolor=MUTED, spikethickness=1)
fig.update_yaxes(gridcolor=BORDER)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# =========================================================
# THREE MODULE CARDS
# =========================================================

st.markdown("#### Module signals")
m1, m2, m3 = st.columns(3)

with m1:
    c = colour_for(prediction_direction)
    st.markdown(
        f"""<div class="card">
        <div class="sig-title">1 · LSTM forecast</div>
        <div style="margin:8px 0">{pill(prediction_direction, c)}</div>
        <div class="rowline"><span class="k">Predicted return</span>
            <span class="mono" style="color:{colour_for(pred['predicted_return'])}">
            {pred['predicted_return']*100:+.2f}%</span></div>
        <div class="rowline"><span class="k">Predicted close</span>
            <span class="mono">₹{predicted_price:,.2f}</span></div>
        <div class="rowline"><span class="k">Signal</span>
            <span class="mono">{recommendation_signal}</span></div>
        </div>""", unsafe_allow_html=True)

with m2:
    c = colour_for(technical_signal)
    st.markdown(
        f"""<div class="card">
        <div class="sig-title">2 · Technical analysis</div>
        <div style="margin:8px 0">{pill(technical_signal, c)}</div>
        <div class="rowline"><span class="k">RSI (14)</span>
            <span class="mono">{technical_result.get('rsi', 0):.1f}</span></div>
        <div class="rowline"><span class="k">Trend (SMA20/50)</span>
            <span class="mono">{technical_result.get('trend','-')}</span></div>
        <div class="rowline"><span class="k">MACD</span>
            <span class="mono">{technical_result.get('macd_trend','-')}</span></div>
        <div class="rowline"><span class="k">Volatility</span>
            <span class="mono">{technical_result.get('volatility_level','-')}</span></div>
        </div>""", unsafe_allow_html=True)

with m3:
    c = colour_for(news_sentiment)
    st.markdown(
        f"""<div class="card">
        <div class="sig-title">3 · News sentiment</div>
        <div style="margin:8px 0">{pill(news_sentiment, c)}</div>
        <div class="rowline"><span class="k">Headlines</span>
            <span class="mono">{news_result.get('total_news', 0)}</span></div>
        <div class="rowline"><span class="k">Pos / Neu / Neg</span>
            <span class="mono">{news_result.get('positive',0)} / {news_result.get('neutral',0)} / {news_result.get('negative',0)}</span></div>
        <div class="rowline"><span class="k">Score</span>
            <span class="mono">{news_result.get('sentiment_score',0):+.2f}</span></div>
        <div class="rowline"><span class="k">Most recent</span>
            <span class="mono">{news_result.get('most_recent','n/a')}</span></div>
        </div>""", unsafe_allow_html=True)


# =========================================================
# CONSENSUS  +  FINAL CALL
# =========================================================

st.markdown("#### Coordinated decision")
c1, c2 = st.columns([2, 3])

with c1:
    conf_pct = max(0, min(100, int(confidence_score)))
    st.markdown(
        f"""<div class="card">
        <div class="rowline"><span class="k">Coordinated signal</span>
            <span class="mono" style="color:{colour_for(coordination_result['coordinated_signal'])}">
            {coordination_result['coordinated_signal']}</span></div>
        <div class="rowline"><span class="k">Module agreement</span>
            <span class="mono">{agreement}</span></div>
        <div class="rowline"><span class="k">Bullish / Bearish / Neutral</span>
            <span class="mono">{coordination_result['bullish_modules']} / {coordination_result['bearish_modules']} / {coordination_result['neutral_modules']}</span></div>
        <div class="rowline"><span class="k">Confidence</span>
            <span class="mono">{confidence_score:.0f}%  ·  {confidence_level}</span></div>
        <div class="rowline"><span class="k">Risk</span>
            <span class="mono" style="color:{BEAR if risk_level=='HIGH' else FLAT}">
            {risk_score}/10 · {risk_level}</span></div>
        </div>""", unsafe_allow_html=True)
    st.progress(conf_pct, text=f"confidence {conf_pct}%")

with c2:
    st.markdown(
        f"""<div class="card" style="border-left:3px solid {rec_colour}">
        <div class="sig-title">Final recommendation</div>
        <div style="font-size:1.9rem;font-weight:800;color:{rec_colour};margin:6px 0">{recommendation}</div>
        <div style="color:{TEXT};font-size:.9rem;line-height:1.6">
            {final_result.get('explanation','')}</div>
        </div>""", unsafe_allow_html=True)

if risk_result.get("risk_factors"):
    with st.expander("Risk factors"):
        for rf in risk_result["risk_factors"]:
            st.markdown(f"- {rf}")
if technical_result.get("signals"):
    with st.expander("Technical detail"):
        for s in technical_result["signals"]:
            st.markdown(f"- {s}")


# =========================================================
# MODEL ACCURACY  (hold-out back-test)
# =========================================================

st.markdown("#### Model accuracy · hold-out back-test")

if model_metrics:
    a1, a2, a3, a4 = st.columns(4)
    da   = model_metrics.get("directional_accuracy")
    ba   = model_metrics.get("baseline_always_up")
    ca   = model_metrics.get("conviction_accuracy")
    edge = (da - ba) if (da is not None and ba is not None) else None

    kpi(a1, "High-conviction acc.", f"{ca:.1f}%" if ca is not None else "N/A",
        f"top {model_metrics.get('conviction_coverage', 33):.0f}% of days",
        BULL if (ca or 0) >= 52 else FLAT)
    kpi(a2, "All-days direction", f"{da:.1f}%" if da is not None else "N/A",
        f"{edge:+.1f} pts vs naive" if edge is not None else "",
        BULL if (edge or 0) > 0 else BEAR if (edge or 0) < 0 else FLAT)
    kpi(a3, "Price error (MAPE)", f"{model_metrics.get('mape', float('nan')):.2f}%",
        f"₹{model_metrics.get('mae', 0):,.0f} avg abs error", TEXT)
    kpi(a4, "Back-test window", f"{model_metrics.get('n_samples', 0):,} days",
        f"trained {model_metrics.get('trained_at', '—')}", TEXT)

    st.caption(
        f"Up-day hit {model_metrics.get('up_accuracy','–')}% · "
        f"down-day hit {model_metrics.get('down_accuracy','–')}% · "
        f"persistence baseline {model_metrics.get('baseline_persistence','–')}%. "
        "High-conviction accuracy is the hit-rate on the third of days with the largest "
        "predicted move — the number to act on. Predicting every day's direction is "
        "near-random by nature; MAPE shows the price forecast is close in magnitude."
    )
else:
    st.info("No saved metrics for this model. Run `python training/retrain_all.py`.")


# =========================================================
# NEWS FEED
# =========================================================

st.markdown("#### Live news feed")
age = news_result.get("newest_age_days")
st.caption(
    f"\U0001f7e2 fetched {news_result.get('fetched_at','n/a')} · "
    f"most recent {news_result.get('most_recent','n/a')}"
    + (f" ({age:.1f} days old)" if isinstance(age, (int, float)) else "")
    + f" · {news_result.get('total_news', 0)} headlines scored"
)

for item in news_result.get("news", []):
    s = normalize_sentiment(item.get("sentiment", "NEUTRAL"))
    sc = colour_for(s)
    link = item.get("link", "")
    title = item.get("title", "")
    title_html = f'<a href="{link}" target="_blank" style="color:{TEXT};text-decoration:none">{title}</a>' if link else title
    st.markdown(
        f"""<div class="news" style="border-left-color:{sc}">
            <div class="h">{title_html}</div>
            <div class="m mono">{item.get('publisher','Unknown')} · {item.get('published_str','')}
                &nbsp;·&nbsp; <span style="color:{sc}">{s}</span> ({item.get('score',0):+d})</div>
        </div>""",
        unsafe_allow_html=True,
    )

if not news_result.get("news"):
    st.info("No news headlines were returned for this instrument.")


st.markdown(
    f"<div class='kpi-sub' style='margin-top:22px'>AlphaStream is a research/education tool. "
    f"Forecasts are model output, not investment advice. Markets carry risk of loss.</div>",
    unsafe_allow_html=True,
)
