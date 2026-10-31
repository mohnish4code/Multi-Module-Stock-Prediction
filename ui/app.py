"""AlphaStream — Streamlit front-end.  Run:  streamlit run ui/app.py"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="AlphaStream · AI Equity Terminal", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

import client  # noqa: E402
from theme import ACCENT, BEAR, BG, BORDER, BULL, FLAT, MUTED, TEXT, apply_theme, colour_for, kpi, pill  # noqa: E402

apply_theme()

MODEL_LABELS = {
    "buy_and_hold": "Buy & hold", "xgboost": "XGBoost", "lstm": "LSTM", "transformer": "Transformer",
    "arima": "ARIMA", "majority": "Majority direction", "persistence": "Persistence", "random_walk": "Random walk",
    "xgboost+news": "XGBoost + news", "lstm+news": "LSTM + news",
}


# ---------------------------------------------------------------- helpers

def ist_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC) + dt.timedelta(hours=5, minutes=30)


def market_state() -> tuple[str, str]:
    now = ist_now()
    minutes = now.hour * 60 + now.minute
    return ("OPEN", BULL) if now.weekday() < 5 and 555 <= minutes <= 930 else ("CLOSED", BEAR)


@st.cache_data(ttl=300, show_spinner=False)
def cached_quotes(symbols: tuple[str, ...]) -> dict[str, Any]:
    try:
        return client.quotes(list(symbols))
    except Exception:  # noqa: BLE001 - the tape is cosmetic
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def cached_backtest(ticker: str) -> dict[str, Any]:
    return client.backtest(ticker)


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "n/a"
    return dt.datetime.fromisoformat(iso).astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime(
        "%d %b %Y, %H:%M")


def dark_layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11, color=TEXT)))
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)


# ---------------------------------------------------------------- chrome

try:
    TICKERS = client.tickers()
except client.ClientError as exc:
    st.error(str(exc))
    st.caption(f"Start it with `uvicorn src.api.main:app` or `docker compose up` (API_URL={client.API_URL}).")
    st.stop()
symbols = list(TICKERS)
quotes = cached_quotes(tuple(symbols))
state, state_colour = market_state()

st.markdown(
    f"""<div class="topbar">
        <div><span class="brand">◆ ALPHASTREAM</span><small>AI EQUITY TERMINAL</small></div>
        <div class="clock mono"><span class="dot" style="background:{state_colour}"></span>NSE {state}
        &nbsp;|&nbsp; {ist_now().strftime('%d %b %Y  %H:%M')} IST</div></div>""",
    unsafe_allow_html=True,
)

cells = []
for t in symbols:
    q, sym = quotes.get(t), t.removesuffix(".NS")
    if not q:
        cells.append(f'<span class="cell"><span class="sym">{sym}</span><span class="px">--</span></span>')
        continue
    arrow = "▲" if q["change_pct"] > 0 else "▼" if q["change_pct"] < 0 else "•"
    cells.append(f'<span class="cell mono"><span class="sym">{sym}</span><span class="px">{q["price"]:,.2f}</span>'
                 f'<span style="color:{colour_for(q["change_pct"])}">{arrow} {q["change_pct"]:+.2f}%</span></span>')
st.markdown(f'<div class="tape">{"".join(cells)}</div>', unsafe_allow_html=True)


def _fmt(t: str) -> str:
    q, sym = quotes.get(t), t.removesuffix(".NS")
    return f"{sym:<11} {q['price']:>9,.0f}  {q['change_pct']:+.2f}%" if q else f"{sym:<11}      --"


st.sidebar.markdown("### Instruments")
ticker = st.sidebar.radio("instrument", symbols, format_func=_fmt, label_visibility="collapsed")
company = TICKERS[ticker]["name"]
st.sidebar.markdown("---")
run = st.sidebar.button("RUN AI ANALYSIS", width="stretch")
st.sidebar.markdown(
    f"""<div style="color:{MUTED};font-size:.76rem;line-height:1.7;margin-top:10px">
    <b style="color:{TEXT}">Pipeline</b><br>Model forecast · Technical read · News sentiment<br>
    → coordination → agreement → risk → decision<br><br>
    <b style="color:{TEXT}">Data</b><br>Yahoo Finance EOD · Google News RSS<br>Horizon: next trading day</div>""",
    unsafe_allow_html=True,
)

q = quotes.get(ticker)
head_l, head_r = st.columns([3, 2])
head_l.markdown(f"## {ticker.removesuffix('.NS')} <span style='color:{MUTED};font-size:1rem;font-weight:500'>"
                f"· {company}</span>", unsafe_allow_html=True)
if q:
    head_r.markdown(
        f"""<div style="text-align:right"><span class="mono" style="font-size:1.8rem;font-weight:750">
        {q['price']:,.2f}</span><span class="mono" style="color:{colour_for(q['change'])};font-size:1rem;
        margin-left:8px">{q['change']:+,.2f} ({q['change_pct']:+.2f}%)</span>
        <div class="kpi-sub">last EOD close</div></div>""",
        unsafe_allow_html=True,
    )

tab_analysis, tab_backtest = st.tabs(["Analysis", "Backtest vs baselines"])


# ---------------------------------------------------------------- backtest tab

def render_backtest(bt: dict[str, Any]) -> None:
    metrics = pd.DataFrame(bt["metrics"])
    allf = metrics[metrics["fold"] == "all"].set_index("model")

    st.markdown("#### Walk-forward equity · long/flat strategy, net of costs")
    eq = bt["equity"]
    fig = go.Figure()
    for name, curve in eq["curves"].items():
        if name == "random_walk":
            continue
        is_bh = name == "buy_and_hold"
        fig.add_trace(go.Scatter(x=eq["dates"], y=curve, name=MODEL_LABELS.get(name, name),
                                 line=dict(width=2.4 if is_bh else 1.3, color=TEXT if is_bh else None)))
    dark_layout(fig, 420)
    fig.update_yaxes(title="Growth of ₹1")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"Out-of-sample {eq['dates'][0]} → {eq['dates'][-1]}; every model retrained per fold "
               f"(expanding window). Long next day when the forecast is positive, else cash; "
               f"{bt['cost_per_side']:.2%} cost per side.")

    st.markdown("#### Model vs baselines")
    table = pd.DataFrame({
        "Directional acc. (%)": allf["dir_acc"],
        "p-value vs 50%": allf["p_vs_50"],
        "MAPE (%)": allf["mape"],
        "RMSE (₹)": allf["rmse"],
        "Sharpe (net)": allf["strat_sharpe"],
        "CAGR": allf["strat_cagr"],
        "Max drawdown": allf["strat_max_dd"],
        "Time in market": allf["exposure"],
    })
    bh = allf.iloc[0]
    table.loc["buy_and_hold"] = [None, None, None, None, bh["bh_sharpe"], bh["bh_cagr"], bh["bh_max_dd"], 1.0]
    table.index = [MODEL_LABELS.get(i, i) for i in table.index]
    st.dataframe(
        table.style.format({
            "Directional acc. (%)": "{:.1f}", "p-value vs 50%": "{:.3f}", "MAPE (%)": "{:.3f}", "RMSE (₹)": "{:,.1f}",
            "Sharpe (net)": "{:.2f}", "CAGR": "{:.1%}", "Max drawdown": "{:.1%}", "Time in market": "{:.0%}",
        }, na_rep="—"),
        width="stretch",
    )
    st.caption("Random walk = tomorrow's price equals today's (no directional call). Persistence repeats "
               "yesterday's direction. Majority always predicts the training set's most common direction.")

    st.markdown("#### Directional accuracy by fold")
    folds = metrics[(metrics["fold"] != "all")].dropna(subset=["dir_acc"])
    fig = go.Figure()
    for name, g in folds.groupby("model"):
        fig.add_trace(go.Bar(x=g["test_start"].astype(str), y=g["dir_acc"], name=MODEL_LABELS.get(name, name)))
    fig.add_hline(y=50, line_dash="dash", line_color=MUTED)
    dark_layout(fig, 320)
    fig.update_yaxes(range=[40, 60], title="%")
    fig.update_xaxes(title="Test block start")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


with tab_backtest:
    try:
        render_backtest(cached_backtest(ticker))
    except client.ClientError as exc:
        st.info(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load backtest results: {exc}")


# ---------------------------------------------------------------- analysis tab

def render_analysis(a: dict[str, Any], bt: dict[str, Any] | None) -> None:
    f, t, n, d = a["forecast"], a["technical"], a["news"], a["decision"]
    rec_colour = colour_for(d["recommendation"])

    st.markdown(f"<div style='margin:6px 0 14px'>SIGNAL &nbsp; {pill(d['recommendation'], rec_colour)} "
                f"<span class='kpi-sub'>&nbsp; coordinated: {d['coordinated_signal']} · "
                f"{d['agreement'].lower()} agreement · data as of {f['as_of']}</span></div>",
                unsafe_allow_html=True)

    serving = None
    if bt:
        rows = [m for m in bt["metrics"] if m["model"] == f["model"] and m["fold"] == "all"]
        serving = rows[0] if rows else None

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Predicted close", f"₹{f['predicted_price']:,.2f}", f"{f['model']} · next session")
    kpi(k2, "Expected move", f"{f['predicted_change_pct']:+.2f}%", f"₹{f['predicted_change']:+,.2f} vs last close",
        colour_for(f["predicted_change"]))
    if serving:
        kpi(k3, "Backtest direction acc.", f"{serving['dir_acc']:.1f}%",
            f"p = {serving['p_vs_50']:.2f} vs coin flip · {serving['n']:,} days",
            BULL if serving["p_vs_50"] < 0.05 else FLAT)
    else:
        kpi(k3, "Backtest direction acc.", "N/A", "run backtest.py")
    kpi(k4, "Risk level", d["risk_level"], f"score {d['risk_score']}/10",
        BEAR if d["risk_level"] == "HIGH" else FLAT if d["risk_level"] == "MEDIUM" else BULL)

    st.markdown("#### Price action & forecast")
    h = pd.DataFrame(a["history"])
    h["date"] = pd.to_datetime(h["date"])
    nxt = h["date"].iloc[-1] + pd.tseries.offsets.BDay(1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"],
                                 name="OHLC", increasing_line_color=BULL, decreasing_line_color=BEAR,
                                 increasing_fillcolor=BULL, decreasing_fillcolor=BEAR, line=dict(width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=h["bb_upper"], line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=h["bb_lower"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(59,130,246,0.08)", showlegend=False, hoverinfo="skip"), row=1, col=1)
    for col, colour, label in (("sma_20", "#f0b90b", "SMA 20"), ("ema_20", "#8b5cf6", "EMA 20")):
        fig.add_trace(go.Scatter(x=h["date"], y=h[col], name=label, line=dict(color=colour, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[h["date"].iloc[-1], nxt], y=[h["close"].iloc[-1], f["predicted_price"]],
                             name="Forecast", mode="lines+markers", line=dict(color=ACCENT, width=2, dash="dot"),
                             marker=dict(size=9, color=ACCENT, symbol="diamond")), row=1, col=1)
    vol_colours = [BULL if c >= o else BEAR for o, c in zip(h["open"], h["close"], strict=True)]
    fig.add_trace(go.Bar(x=h["date"], y=h["volume"], marker_color=vol_colours, opacity=0.45, name="Volume"),
                  row=2, col=1)
    dark_layout(fig, 520)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Module signals")
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="card"><div class="sig-title">1 · Model forecast ({f['model']})</div>
        <div style="margin:8px 0">{pill(f['direction'], colour_for(f['direction']))}</div>
        <div class="rowline"><span class="k">Predicted return</span><span class="mono"
            style="color:{colour_for(f['predicted_return'])}">{f['predicted_return'] * 100:+.3f}%</span></div>
        <div class="rowline"><span class="k">Predicted close</span><span class="mono">
            ₹{f['predicted_price']:,.2f}</span></div>
        <div class="rowline"><span class="k">Trained</span><span class="mono">
            {f['model_trained_at'][:10]}</span></div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="card"><div class="sig-title">2 · Technical analysis</div>
        <div style="margin:8px 0">{pill(t['technical_signal'], colour_for(t['technical_signal']))}</div>
        <div class="rowline"><span class="k">RSI (14)</span><span class="mono">{t['rsi']:.1f}</span></div>
        <div class="rowline"><span class="k">Trend (SMA20/50)</span><span class="mono">{t['trend']}</span></div>
        <div class="rowline"><span class="k">MACD</span><span class="mono">{t['macd_trend']}</span></div>
        <div class="rowline"><span class="k">Volatility</span><span class="mono">{t['volatility_level']}</span>
        </div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="card"><div class="sig-title">3 · News sentiment ({n['method']})</div>
        <div style="margin:8px 0">{pill(n['sentiment_label'], colour_for(n['sentiment_label']))}</div>
        <div class="rowline"><span class="k">Headlines</span><span class="mono">{n['total_news']}</span></div>
        <div class="rowline"><span class="k">Pos / Neu / Neg</span><span class="mono">
            {n['positive']} / {n['neutral']} / {n['negative']}</span></div>
        <div class="rowline"><span class="k">Score</span><span class="mono">{n['sentiment_score']:+.2f}</span></div>
        <div class="rowline"><span class="k">Most recent</span><span class="mono">{fmt_date(n['most_recent'])}</span>
        </div></div>""", unsafe_allow_html=True)

    st.markdown("#### Coordinated decision")
    c1, c2 = st.columns([2, 3])
    c1.markdown(f"""<div class="card">
        <div class="rowline"><span class="k">Coordinated signal</span><span class="mono"
            style="color:{colour_for(d['coordinated_signal'])}">{d['coordinated_signal']}</span></div>
        <div class="rowline"><span class="k">Module agreement</span><span class="mono">{d['agreement']}</span></div>
        <div class="rowline"><span class="k">Bullish / Bearish / Neutral</span><span class="mono">
            {d['bullish_modules']} / {d['bearish_modules']} / {d['neutral_modules']}</span></div>
        <div class="rowline"><span class="k">Agreement score</span><span class="mono">
            {d['confidence_score']}/100 · {d['confidence_level']}</span></div>
        <div class="rowline"><span class="k">Risk</span><span class="mono"
            style="color:{BEAR if d['risk_level'] == 'HIGH' else FLAT}">{d['risk_score']}/10 · {d['risk_level']}
            </span></div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="card" style="border-left:3px solid {rec_colour}">
        <div class="sig-title">Final recommendation</div>
        <div style="font-size:1.9rem;font-weight:800;color:{rec_colour};margin:6px 0">{d['recommendation']}</div>
        <div style="color:{TEXT};font-size:.9rem;line-height:1.6">{d['explanation']}</div></div>""",
                unsafe_allow_html=True)
    st.caption("The agreement and risk scores are rule-based heuristics, not calibrated probabilities. "
               "See the Backtest tab for how the forecast has actually performed.")
    if d["risk_factors"]:
        with st.expander("Risk factors"):
            st.markdown("\n".join(f"- {r}" for r in d["risk_factors"]))
    with st.expander("Technical detail"):
        st.markdown("\n".join(f"- {s}" for s in t["signals"]))

    st.markdown("#### News feed")
    age = n.get("newest_age_days")
    st.caption(f"fetched {fmt_date(n.get('fetched_at'))} · most recent {fmt_date(n.get('most_recent'))}"
               + (f" ({age:.1f} days old)" if isinstance(age, int | float) else "")
               + f" · {n['total_news']} headlines scored")
    for item in n["news"]:
        sc = colour_for(item["sentiment"])
        title = (f'<a href="{item["link"]}" target="_blank" style="color:{TEXT};text-decoration:none">'
                 f'{item["title"]}</a>') if item.get("link") else item["title"]
        st.markdown(f"""<div class="news" style="border-left-color:{sc}"><div class="h">{title}</div>
            <div class="m mono">{item.get('publisher', 'Unknown')} · {fmt_date(item.get('published'))} &nbsp;·&nbsp;
            <span style="color:{sc}">{item['sentiment']}</span> ({item['score']:+.2f})</div></div>""",
                    unsafe_allow_html=True)
    if not n["news"]:
        st.info("No news headlines were returned for this instrument.")


with tab_analysis:
    key = f"analysis:{ticker}"
    if run:
        with st.spinner("Fetching data, forecasting, scoring news..."):
            try:
                st.session_state[key] = client.analysis(ticker)
            except client.ClientError as exc:
                st.warning(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error("The analysis could not be completed.")
                st.exception(exc)

    if key in st.session_state:
        try:
            bt_data = cached_backtest(ticker)
        except Exception:  # noqa: BLE001
            bt_data = None
        render_analysis(st.session_state[key], bt_data)
    else:
        st.markdown(
            f"""<div class="card" style="margin-top:14px;text-align:center;padding:46px">
            <div style="font-size:2rem">◆</div>
            <div style="font-size:1.05rem;font-weight:650;margin-top:8px">{company} selected</div>
            <div class="kpi-sub" style="margin-top:6px">Press <b style="color:{TEXT}">RUN AI ANALYSIS</b> for the
            next-day forecast, technical read, news sentiment and the coordinated call. The
            <b style="color:{TEXT}">Backtest</b> tab shows how each model has performed out-of-sample.</div></div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div class='kpi-sub' style='margin-top:22px'>AlphaStream is a research/education tool. "
            "Forecasts are model output, not investment advice.</div>", unsafe_allow_html=True)
