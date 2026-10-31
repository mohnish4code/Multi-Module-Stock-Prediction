"""Terminal theme: colours, global CSS and small HTML helpers."""

from __future__ import annotations

import streamlit as st

BG = "#0a0e17"
SURFACE = "#111725"
SURFACE_2 = "#161d2e"
BORDER = "#1f2a3d"
TEXT = "#dce3ef"
MUTED = "#7d8ba3"
BULL = "#16c784"
BEAR = "#ea3943"
FLAT = "#8b97ab"
ACCENT = "#3b82f6"

CSS = f"""
<style>
#MainMenu, header, footer {{ visibility: hidden; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
.block-container {{ padding: 0.6rem 1.4rem 3rem; max-width: 1550px; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}
h1, h2, h3, h4 {{ color: #f2f5fa; letter-spacing: .2px; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
.mono {{ font-variant-numeric: tabular-nums;
         font-family: "SF Mono","Roboto Mono",ui-monospace,Consolas,monospace; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between;
           padding:10px 16px; border:1px solid {BORDER}; border-radius:12px;
           background:linear-gradient(90deg,{SURFACE},{SURFACE_2}); margin-bottom:10px; }}
.brand {{ font-weight:800; font-size:1.05rem; letter-spacing:1.5px; color:#f4f7fc; }}
.brand small {{ color:{MUTED}; font-weight:600; letter-spacing:2px; margin-left:10px; font-size:.7rem; }}
.clock {{ color:{MUTED}; font-size:.82rem; }}
.dot {{ height:8px; width:8px; border-radius:50%; display:inline-block; margin-right:6px; }}
.tape {{ display:flex; gap:22px; overflow-x:auto; padding:9px 14px; border:1px solid {BORDER};
         border-radius:12px; background:{SURFACE}; margin-bottom:14px; scrollbar-width:none; }}
.tape::-webkit-scrollbar {{ display:none; }}
.tape .cell {{ white-space:nowrap; font-size:.82rem; }}
.tape .sym {{ color:{TEXT}; font-weight:700; margin-right:8px; }}
.tape .px  {{ color:{MUTED}; margin-right:6px; }}
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
.news {{ border:1px solid {BORDER}; border-left:3px solid {BORDER};
         border-radius:10px; padding:10px 14px; margin-bottom:8px; background:{SURFACE}; }}
.news .h {{ font-weight:600; font-size:.92rem; color:{TEXT}; }}
.news .m {{ color:{MUTED}; font-size:.75rem; margin-top:3px; }}
[data-testid="stSidebar"] .stRadio > div {{ gap:2px; }}
[data-testid="stSidebar"] .stRadio label {{
    border:1px solid transparent; border-radius:9px; padding:7px 9px; width:100%; transition:.12s; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background:{SURFACE_2}; }}
.stButton > button {{ width:100%; border-radius:10px; font-weight:750; letter-spacing:.6px;
    background:{ACCENT}; color:#fff; border:none; padding:.62rem; }}
.stButton > button:hover {{ background:#2f6fe0; color:#fff; }}
div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:12px; background:{SURFACE}; }}
hr {{ border-color:{BORDER}; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def colour_for(value: object) -> str:
    """Green/red/grey for a signed number or a signal word."""
    if isinstance(value, int | float):
        return BULL if value > 0 else BEAR if value < 0 else FLAT
    return {"BULLISH": BULL, "BUY": BULL, "POSITIVE": BULL,
            "BEARISH": BEAR, "SELL": BEAR, "NEGATIVE": BEAR}.get(str(value).upper().strip(), FLAT)


def pill(text: str, colour: str) -> str:
    return (f'<span class="pill" style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}55">{text}</span>')


def kpi(col, label: str, value: str, sub: str = "", value_colour: str = TEXT) -> None:
    col.markdown(
        f"""<div class="card"><div class="kpi-label">{label}</div>
        <div class="kpi-value mono" style="color:{value_colour}">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )
