"""
E-Waste Classification Model
Streamlit application using the existing TensorFlow model.
Run: python -m streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import logging
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ewaste_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
INFO_PATH = BASE_DIR / "ewaste_info.json"

IMAGE_SIZE = (224, 224)
TOP_K = 5
HIGH_CONFIDENCE = 80.0
MODERATE_CONFIDENCE = 60.0
CLOSE_CALL_MARGIN = 10.0
MAX_UPLOAD_MB = 15

# "raw" | "rescale" | "mobilenet_v2" — must match training
PREPROCESSING = "raw"

COLOR_HIGH = "#5fd1c0"
COLOR_MODERATE = "#f0c35c"
COLOR_LOW = "#ef7b6c"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ewaste_classifier")

st.set_page_config(
    page_title="E-Waste Classification Model",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# Theme: deep circuit-board teal with copper accents
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        color-scheme: dark;
        --bg: #0a1720;
        --bg-2: #0c1a22;
        --panel: #122630;
        --panel-hi: #17303c;
        --line: #264553;
        --line-hi: #3b6272;
        --text: #e9f2f3;
        --body: #cfe0e2;
        --muted: #94afb6;
        --copper: #e39a55;
        --copper-hi: #f3b977;
        --teal: #5fd1c0;
        --gold: #f0c35c;
        --coral: #ef7b6c;
        --mint: #8fdca8;
    }

    .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-icons) {
        font-family: 'Figtree', 'Segoe UI', system-ui, sans-serif;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .head-title, .sec, .panel h4, .cat-name, .ring-in b, .rank-badge, .spec-val,
    .panel-title, .stat-val {
        font-family: 'Bricolage Grotesque', 'Figtree', system-ui, sans-serif !important;
    }

    /* ---------- Animated mesh background ---------- */
    @keyframes drift {
        0%   { transform: translate3d(0,0,0) scale(1); }
        50%  { transform: translate3d(2%, -1.5%, 0) scale(1.05); }
        100% { transform: translate3d(0,0,0) scale(1); }
    }
    @keyframes pulse {
        0%,100% { opacity: .55; }
        50%     { opacity: .9; }
    }
    .stApp::before {
        content: "";
        position: fixed; inset: -20%;
        background:
            radial-gradient(closest-side at 20% 20%, rgba(95,209,192,.16), transparent 70%),
            radial-gradient(closest-side at 80% 10%, rgba(227,154,85,.14), transparent 70%),
            radial-gradient(closest-side at 60% 90%, rgba(95,209,192,.10), transparent 70%),
            radial-gradient(closest-side at 10% 80%, rgba(240,195,92,.08), transparent 70%);
        animation: drift 22s ease-in-out infinite;
        z-index: 0; pointer-events: none;
    }
    .stApp {
        background: var(--bg-2);
        color: var(--text);
        position: relative;
    }
    .stApp > * { position: relative; z-index: 1; }

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
        visibility: hidden !important; display: none !important;
    }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }

    @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
    @keyframes grow { from { width: 0; } }
    @keyframes floaty {
        0%,100% { transform: translateY(0); }
        50%     { transform: translateY(-6px); }
    }
    @keyframes shine {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ---------- Hero ---------- */
    .page-head {
        position: relative;
        padding: 34px 34px 30px 34px;
        margin-bottom: 28px;
        border: 1px solid var(--line);
        border-radius: 22px;
        background:
            radial-gradient(900px 260px at 90% -30%, rgba(227,154,85,.18), transparent 60%),
            radial-gradient(700px 260px at -10% 110%, rgba(95,209,192,.14), transparent 60%),
            linear-gradient(180deg, rgba(23,48,60,.85), rgba(18,38,48,.85));
        box-shadow: 0 30px 60px -32px rgba(0,0,0,.75), inset 0 1px 0 rgba(255,255,255,.04);
        overflow: hidden;
        animation: rise .6s ease both;
    }
    .page-head::after {
        content: "";
        position: absolute; left: 34px; bottom: 0;
        width: 140px; height: 4px; border-radius: 4px 4px 0 0;
        background: linear-gradient(90deg, var(--copper), var(--teal));
    }
    .page-head .orb {
        position: absolute; border-radius: 50%; filter: blur(6px); opacity: .55;
        animation: pulse 6s ease-in-out infinite;
        pointer-events: none;
    }
    .page-head .orb.a { top: -30px; right: 60px; width: 140px; height: 140px; background: radial-gradient(closest-side, rgba(227,154,85,.35), transparent); animation-delay: 0s; }
    .page-head .orb.b { bottom: -40px; right: 220px; width: 180px; height: 180px; background: radial-gradient(closest-side, rgba(95,209,192,.28), transparent); animation-delay: 1.5s; }
    .page-head .orb.c { top: 40px; right: 380px; width: 90px; height: 90px; background: radial-gradient(closest-side, rgba(240,195,92,.25), transparent); animation-delay: 3s; }

    .eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: .76rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase;
        color: var(--teal);
        padding: 6px 12px; border-radius: 999px;
        background: rgba(95,209,192,.08);
        border: 1px solid rgba(95,209,192,.25);
        margin-bottom: 16px;
    }
    .eyebrow::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 10px var(--teal); }

    .head-title {
        font-size: 3.1rem; line-height: 1.02; font-weight: 800; letter-spacing: -0.03em;
        background: linear-gradient(120deg, #ffffff 0%, #cfe0e2 45%, var(--copper) 100%);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 12px 0;
    }
    .head-sub { color: var(--body); font-size: 1.06rem; margin: 0 0 18px 0; max-width: 68ch; line-height: 1.6; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
        padding: 6px 13px; border-radius: 999px; font-size: .82rem; font-weight: 500;
        color: var(--body);
        background: rgba(255,255,255,.035);
        border: 1px solid var(--line);
        backdrop-filter: blur(4px);
        transition: border-color .2s, transform .2s;
    }
    .chip:hover { border-color: var(--copper); transform: translateY(-1px); }

    /* ---------- Section headers with icon badge ---------- */
    .sec {
        position: relative;
        display: flex; align-items: center; gap: 12px;
        font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em;
        color: var(--text);
        margin: 2.6rem 0 .5rem 0;
        padding-bottom: 12px;
    }
    .sec::after {
        content: "";
        position: absolute; left: 0; bottom: 0;
        width: 72px; height: 3px; border-radius: 3px;
        background: linear-gradient(90deg, var(--copper), var(--teal));
    }
    .sec .sec-ico {
        display: inline-grid; place-items: center;
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, rgba(227,154,85,.22), rgba(95,209,192,.18));
        border: 1px solid var(--line-hi);
        font-size: 1rem;
    }
    .sec-note { color: var(--muted); font-size: .95rem; margin: .35rem 0 1rem 0; }

    /* ---------- Uploader ---------- */
    div[data-testid="stFileUploader"] section {
        background: linear-gradient(180deg, rgba(23,48,60,.6), rgba(18,38,48,.7));
        border: 1.5px dashed #4a6b78; border-radius: 16px;
        transition: border-color .25s, background .25s;
    }
    div[data-testid="stFileUploader"] section:hover { border-color: var(--copper); background: linear-gradient(180deg, rgba(23,48,60,.85), rgba(18,38,48,.9)); }
    div[data-testid="stFileUploader"] label p { color: var(--text); font-weight: 600; }
    [data-testid="stImage"] { width: 100% !important; }
    [data-testid="stImage"] img {
        width: 100% !important; max-height: 340px; object-fit: contain;
        background: #07121a; border: 1px solid var(--line); border-radius: 16px;
        box-shadow: 0 18px 40px -22px rgba(0,0,0,.8);
    }
    .hint { color: var(--muted); font-size: .88rem; margin: 6px 0 14px 0; }

    /* ---------- Buttons ---------- */
    div.stButton > button, div.stDownloadButton > button {
        width: 100%; min-height: 50px; border-radius: 12px; font-weight: 700;
        letter-spacing: .01em;
        transition: transform .15s, background .15s, box-shadow .2s;
    }
    div.stButton > button:hover:not(:disabled) { transform: translateY(-1px); }
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--copper), var(--copper-hi));
        border: 1px solid var(--copper);
        color: #1b1108;
        box-shadow: 0 14px 28px -14px rgba(227,154,85,.55);
    }
    div.stButton > button[kind="primary"] p,
    div.stButton > button[data-testid="stBaseButton-primary"] p { color: #1b1108; font-weight: 800; }
    div.stButton > button[kind="primary"]:hover:not(:disabled) {
        background: linear-gradient(135deg, var(--copper-hi), #f7c78a);
        box-shadow: 0 18px 36px -16px rgba(227,154,85,.7);
    }
    div.stButton > button:disabled { opacity: .4; }

    /* ---------- Result card ---------- */
    .result {
        position: relative;
        background:
            radial-gradient(600px 160px at 100% 0%, rgba(227,154,85,.10), transparent 60%),
            linear-gradient(180deg, rgba(23,48,60,.75), rgba(18,38,48,.85));
        border: 1px solid var(--line-hi);
        border-radius: 20px;
        padding: 28px 30px;
        overflow: hidden;
        animation: rise .5s ease both;
        box-shadow: 0 30px 60px -30px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.05);
    }
    .result::before {
        content: "";
        position: absolute; inset: 0 0 auto 0; height: 4px;
        background: linear-gradient(90deg, var(--copper), var(--teal), var(--gold), var(--copper));
        background-size: 300% 100%;
        animation: shine 6s linear infinite;
    }
    .rank-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 5px 13px; border-radius: 10px;
        font-size: .78rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
        color: #1b1108;
        background: linear-gradient(135deg, var(--copper), var(--copper-hi));
        margin-bottom: 14px;
        box-shadow: 0 8px 18px -8px rgba(227,154,85,.6);
    }
    .result-top { display: flex; align-items: center; justify-content: space-between; gap: 22px; }
    .result-label { color: var(--muted); font-size: .9rem; margin-bottom: 6px; letter-spacing: .04em; text-transform: uppercase; }
    .cat-name {
        font-size: 2.7rem; line-height: 1.02; font-weight: 800; letter-spacing: -0.025em;
        background: linear-gradient(120deg, #ffffff 0%, #cfe0e2 60%, var(--copper) 100%);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 14px; overflow-wrap: anywhere;
    }
    .pill {
        display: inline-block; padding: 5px 13px; border-radius: 999px;
        font-size: .84rem; font-weight: 700;
        border: 1px solid currentColor;
        margin-right: 6px;
        background: rgba(255,255,255,.02);
    }
    .pill.plain { color: var(--muted); }
    .ring {
        flex: 0 0 auto; width: 122px; height: 122px; border-radius: 50%;
        background: conic-gradient(var(--ringc) calc(var(--p) * 1%), var(--line) 0);
        display: grid; place-items: center;
        box-shadow: 0 0 0 1px var(--line-hi), 0 14px 30px -12px rgba(0,0,0,.6);
        position: relative;
    }
    .ring::after {
        content: ""; position: absolute; inset: -6px; border-radius: 50%;
        background: conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,.08) 40deg, transparent 80deg);
        animation: spin 6s linear infinite;
        pointer-events: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .ring-in {
        width: 96px; height: 96px; border-radius: 50%;
        background: radial-gradient(circle at 50% 40%, #17303c, #0e1f28);
        display: grid; place-items: center; align-content: center;
    }
    .ring-in b { font-size: 1.65rem; line-height: 1; color: var(--text); }
    .ring-in span { font-size: .72rem; color: var(--muted); margin-top: 3px; letter-spacing: .1em; text-transform: uppercase; }
    .result-divider {
        height: 1px; margin: 22px 0 16px 0;
        background: linear-gradient(90deg, transparent, var(--line-hi), transparent);
    }
    .alts-title { color: var(--muted); font-size: .88rem; margin-bottom: 12px; letter-spacing: .08em; text-transform: uppercase; }
    .pred { display: grid; grid-template-columns: 30px minmax(90px, 150px) 1fr 58px; align-items: center; gap: 12px; margin: 10px 0; }
    .pred-rank { color: var(--muted); font-size: .82rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .pred.first .pred-rank { color: var(--copper); }
    .pred.first .pred-name { color: var(--copper); font-weight: 700; }
    .pred-name { color: var(--body); font-size: .95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pred-track { height: 9px; border-radius: 5px; background: #142b36; overflow: hidden; display: block; position: relative; }
    .bar {
        display: block; height: 100%; border-radius: 5px;
        background: linear-gradient(90deg, #476b78, #5f8794);
        animation: grow .8s ease both;
        position: relative;
    }
    .bar.top {
        background: linear-gradient(90deg, var(--copper), var(--copper-hi), var(--gold));
        background-size: 200% 100%;
        animation: grow .8s ease both, shine 3s linear infinite;
    }
    .pred-val { text-align: right; color: var(--muted); font-size: .88rem; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
    .note {
        margin-top: 16px; padding: 13px 16px; border-radius: 12px;
        font-size: .93rem; line-height: 1.55;
        background: rgba(15,32,41,.85);
        border: 1px solid var(--line);
        color: var(--body);
    }
    .note.warn { border-color: #7a6226; background: linear-gradient(180deg, #2a2310, #241f10); }
    .note.low { border-color: #7a3f44; background: linear-gradient(180deg, #2a1b1f, #20141a); }

    /* ---------- Empty state ---------- */
    .empty {
        position: relative;
        background: linear-gradient(180deg, rgba(23,48,60,.55), rgba(18,38,48,.65));
        border: 1.5px dashed var(--line-hi);
        border-radius: 20px;
        padding: 40px 34px;
        min-height: 320px;
        display: flex; flex-direction: column; justify-content: center;
        animation: rise .5s ease both;
    }
    .empty .icon {
        font-size: 2.4rem; margin-bottom: 12px;
        display: inline-grid; place-items: center;
        width: 64px; height: 64px; border-radius: 18px;
        background: linear-gradient(135deg, rgba(227,154,85,.18), rgba(95,209,192,.18));
        border: 1px solid var(--line-hi);
        animation: floaty 4s ease-in-out infinite;
    }
    .empty h3 { margin: 0 0 8px 0; font-size: 1.4rem; color: var(--text); }
    .empty p { margin: 0; color: var(--muted); max-width: 46ch; line-height: 1.6; }

    /* ---------- Panels ---------- */
    .panel-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; animation: rise .5s ease both; }
    .panel-row.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .panel-row.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .panel {
        position: relative;
        background: linear-gradient(180deg, rgba(23,48,60,.7), rgba(18,38,48,.75));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 22px 24px;
        transition: border-color .25s, transform .25s, box-shadow .25s;
        overflow: hidden;
    }
    .panel::before {
        content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
        background: linear-gradient(90deg, rgba(227,154,85,.55), rgba(95,209,192,.55));
        opacity: 0; transition: opacity .25s;
    }
    .panel:hover { border-color: var(--line-hi); transform: translateY(-3px); box-shadow: 0 22px 40px -24px rgba(0,0,0,.7); }
    .panel:hover::before { opacity: 1; }
    .panel h4 {
        display: flex; align-items: center; gap: 9px;
        font-size: 1.05rem; font-weight: 700; margin: 0 0 12px 0; color: var(--text);
        letter-spacing: -0.005em;
    }
    .panel p { margin: 0; color: var(--body); line-height: 1.65; }
    .panel ul { margin: 0; padding-left: 1.1rem; }
    .panel li { margin: .45rem 0; color: var(--body); line-height: 1.6; }
    .panel li::marker { color: var(--copper); }
    .panel.danger { background: linear-gradient(180deg, rgba(46,27,31,.85), rgba(38,22,26,.85)); border-color: #6e3a40; }
    .panel.danger h4 { color: var(--coral); }
    .panel.danger li::marker { color: var(--coral); }
    .panel.danger::before { background: linear-gradient(90deg, var(--coral), #f0a092); opacity: 1; }
    .panel.safe { background: linear-gradient(180deg, rgba(16,41,45,.85), rgba(13,36,42,.85)); border-color: #1f4f4d; }
    .panel.safe h4 { color: var(--teal); }
    .panel.safe li::marker { color: var(--teal); }
    .panel.safe::before { background: linear-gradient(90deg, var(--teal), #8ee5d7); opacity: 1; }
    .panel.plain { background: transparent; border-color: var(--line); }
    .panel.eco { background: linear-gradient(180deg, rgba(20,40,30,.85), rgba(16,35,26,.85)); border-color: #2e5c3f; }
    .panel.eco h4 { color: var(--mint); }
    .panel.eco li::marker { color: var(--mint); }
    .panel.eco::before { background: linear-gradient(90deg, var(--mint), #b7efc4); opacity: 1; }

    /* ---------- Spec grid ---------- */
    .spec-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; animation: rise .5s ease both; }
    .spec {
        position: relative;
        background: linear-gradient(180deg, rgba(23,48,60,.7), rgba(18,38,48,.75));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        transition: border-color .25s, transform .25s;
        overflow: hidden;
    }
    .spec:hover { border-color: var(--copper); transform: translateY(-2px); }
    .spec::after {
        content: ""; position: absolute; right: -40px; top: -40px;
        width: 100px; height: 100px; border-radius: 50%;
        background: radial-gradient(closest-side, rgba(95,209,192,.16), transparent);
        pointer-events: none;
    }
    .spec-key {
        font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
        color: var(--muted); margin-bottom: 8px; font-weight: 700;
    }
    .spec-val { font-size: 1.02rem; font-weight: 700; color: var(--text); line-height: 1.35; overflow-wrap: anywhere; }

    /* ---------- Composition bars ---------- */
    .comp { margin: 10px 0; }
    .comp-row { display: flex; justify-content: space-between; font-size: .9rem; color: var(--body); margin-bottom: 5px; }
    .comp-row span:last-child { font-family: 'JetBrains Mono', monospace; color: var(--muted); }
    .comp-track { height: 9px; border-radius: 5px; background: #142b36; overflow: hidden; }
    .comp-bar {
        height: 100%; border-radius: 5px;
        background: linear-gradient(90deg, var(--teal), var(--copper));
        background-size: 200% 100%;
        animation: grow .8s ease both, shine 3.5s linear infinite;
    }

    /* ---------- Tag chips ---------- */
    .tag-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .tag {
        padding: 6px 12px; border-radius: 10px; font-size: .8rem; font-weight: 600;
        background: rgba(95,209,192,.10);
        border: 1px solid #2e5c5b;
        color: var(--teal);
        transition: transform .2s, background .2s;
    }
    .tag:hover { transform: translateY(-1px); background: rgba(95,209,192,.18); }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] { color: var(--muted); font-weight: 700; letter-spacing: .01em; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
    div[data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, var(--copper), var(--teal)) !important;
        height: 3px !important;
    }
    div[data-baseweb="tab-border"] { background-color: var(--line) !important; }
    div[data-baseweb="tab-panel"] { padding-top: 1.1rem; }

    /* ---------- Stat strip (recovered materials, etc.) ---------- */
    .stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 10px; }
    .stat {
        background: linear-gradient(180deg, rgba(23,48,60,.7), rgba(18,38,48,.75));
        border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px;
        display: flex; align-items: center; gap: 14px;
    }
    .stat-ico {
        width: 40px; height: 40px; border-radius: 12px; flex: 0 0 auto;
        display: grid; place-items: center;
        background: linear-gradient(135deg, rgba(227,154,85,.2), rgba(95,209,192,.2));
        border: 1px solid var(--line-hi);
        font-size: 1.05rem;
    }
    .stat-key { font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
    .stat-val { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-top: 2px; }

    .disclaimer {
        margin-top: 3rem; padding: 18px 22px;
        border: 1px solid var(--line); border-radius: 16px;
        background: linear-gradient(180deg, rgba(23,48,60,.45), rgba(18,38,48,.55));
        color: var(--muted); font-size: .88rem; line-height: 1.65; max-width: 95ch;
    }

    @media (max-width: 900px) {
        .spec-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .panel-row.three, .panel-row.four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
        .head-title, .cat-name { font-size: 2rem; }
        .panel-row, .panel-row.three, .panel-row.four { grid-template-columns: 1fr; }
        .result { padding: 20px; }
        .ring { width: 100px; height: 100px; }
        .ring-in { width: 78px; height: 78px; }
        .pred { grid-template-columns: 26px 90px 1fr 52px; gap: 8px; }
        .spec-grid { grid-template-columns: 1fr; }
        .page-head { padding: 24px 22px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Knowledge base (unchanged — every field is preserved for future use)
# ---------------------------------------------------------------------
DEFAULT_INFO: dict[str, dict[str, Any]] = {
    "Battery": {
        "category": "Energy Storage / Power Source",
        "alsoKnownAs": ["Cell", "Battery Pack", "Accumulator", "Power Cell"],
        "description": "An electrochemical device that stores energy and supplies electrical power to electronic devices, appliances and portable equipment. Batteries convert chemical energy into electrical energy through reversible electrochemical reactions.",
        "uses": [
            "Powering mobile phones, laptops and wireless accessories.",
            "Providing backup power to electronic equipment.",
            "Operating portable tools and electronic devices.",
            "Storing energy in compatible energy-storage systems.",
            "Powering electric vehicles and hybrid transport.",
            "Supporting renewable-energy grid storage.",
        ],
        "materials": [
            "Lithium compounds and graphite",
            "Copper and aluminium",
            "Nickel and other metals, depending on chemistry",
            "Lead and sulfuric acid in lead-acid batteries",
            "Steel, plastics and insulating materials",
            "Cobalt, manganese and phosphate in Li-ion cathodes",
            "Electrolyte salts and organic solvents",
        ],
        "composition": {
            "Cathode": "25–35%",
            "Anode": "15–25%",
            "Electrolyte": "10–15%",
            "Separator": "3–5%",
            "Casing": "20–30%",
            "Other metals": "5–15%",
        },
        "components": [
            "Electrochemical cells", "Positive electrode", "Negative electrode",
            "Electrolyte", "Separator", "Protective casing", "Electrical terminals",
            "Battery management system in applicable packs",
            "Thermal protection devices", "Charge controller circuitry",
        ],
        "hazards": [
            "Short circuits can cause overheating and fire.",
            "Damaged lithium-ion batteries may experience thermal runaway.",
            "Leaking electrolytes may cause chemical injuries.",
            "Some batteries contain lead, cadmium or other hazardous substances.",
            "Swollen batteries may rupture violently.",
            "Improper charging may cause explosion.",
        ],
        "safety": [
            "Never puncture, crush, burn or short-circuit a battery.",
            "Do not open sealed battery packs.",
            "Keep batteries away from heat and flames.",
            "Do not handle leaking, swollen or smoking batteries unnecessarily.",
            "Seek qualified assistance for damaged batteries.",
            "Store batteries in a cool, dry, non-conductive container.",
            "Use only approved chargers matching battery specifications.",
        ],
        "reusable": [
            "Batteries professionally tested and confirmed safe",
            "Compatible battery packs with acceptable capacity",
            "Battery materials recovered by authorized recyclers",
            "Second-life batteries for stationary storage",
        ],
        "repair": [
            "Have battery health checked by a qualified technician.",
            "Replace defective batteries with compatible approved replacements.",
            "Do not attempt to repair sealed battery cells yourself.",
            "Recalibrate battery management systems where supported.",
        ],
        "recommendation": [
            "Use battery take-back programs.",
            "Send unusable batteries to authorized battery recyclers.",
            "Keep batteries separate from general waste.",
            "Prefer rechargeable over single-use where practical.",
            "Support extended producer responsibility schemes.",
        ],
        "disposal": [
            "Never burn batteries.",
            "Do not put loose batteries in ordinary waste where separate collection is available.",
            "Follow the collection facility's instructions for damaged batteries.",
            "Tape terminals of lithium batteries before transport.",
            "Use fire-safe containers for damaged cells.",
        ],
        "recyclingProcess": [
            "Collection and sorting by chemistry",
            "Discharge and safe dismantling",
            "Mechanical separation (shredding, density separation)",
            "Pyrometallurgical smelting (recover Co, Ni, Cu)",
            "Hydrometallurgical leaching (recover Li, Co, Ni, Mn)",
            "Refining into new battery-grade materials",
        ],
        "recyclingValue": {
            "Lithium": "High demand, rising value",
            "Cobalt": "Very high value",
            "Nickel": "High value",
            "Copper": "High value",
            "Aluminium": "Moderate value",
            "Lead (Pb-acid)": "High recovery rate (>95%)",
        },
        "recoveryRate": "50–95% depending on chemistry and process",
        "environmentalImpact": [
            "Soil and groundwater contamination from heavy metals",
            "Toxic fumes if burned",
            "Fire risk in landfills",
            "Resource depletion if not recycled",
        ],
        "co2Savings": "Recycling Li-ion saves ~70% CO₂ vs. virgin mining",
        "certifications": ["UN38.3", "IEC 62133", "UL 2054", "CE", "RoHS"],
        "regulations": [
            "EU Battery Regulation 2023/1542",
            "US EPA Universal Waste Rules",
            "Basel Convention (transboundary movement)",
        ],
        "lifespan": "2–10 years depending on chemistry and usage",
        "energyRating": "Varies: 1.5V–400V+ depending on type",
        "commonFaults": [
            "Capacity fade", "Swelling", "Internal short circuit",
            "BMS failure", "Terminal corrosion",
        ],
        "didYouKnow": "One lithium-ion battery can contaminate up to 600,000 litres of water if improperly disposed.",
    },

    "Keyboard": {
        "category": "Computer Peripheral / Input Device",
        "alsoKnownAs": ["Input Board", "Keypad", "QWERTY Board"],
        "description": "A computer input device used to enter text, numbers, symbols and commands. Keyboards may be wired or wireless and may use membrane or mechanical switches. Modern keyboards may include backlighting, macro keys and multimedia controls.",
        "uses": [
            "Typing documents and emails",
            "Programming and data entry",
            "Gaming and computer navigation",
            "Using keyboard shortcuts",
            "Online education and office work",
            "Accessibility input for users with disabilities",
        ],
        "materials": [
            "ABS plastic and other polymers",
            "Copper wiring and conductive tracks",
            "Silicone or rubber membrane layers",
            "Fiberglass-based circuit board material",
            "Steel and other metals",
            "Battery materials in wireless models",
            "PBT plastic in premium keycaps",
            "Gold plating on high-end contacts",
        ],
        "composition": {
            "Plastics": "60–75%",
            "Metals": "10–20%",
            "PCB/Electronics": "10–15%",
            "Rubber/Silicone": "5–10%",
        },
        "components": [
            "Keycaps", "Key switches",
            "Membrane sheet or mechanical switch assembly",
            "Printed circuit board (PCB)", "Keyboard controller", "USB cable",
            "Wireless receiver or Bluetooth module", "Battery in wireless models",
            "Indicator LEDs", "Stabilizers for larger keys",
            "Backlight LEDs in illuminated models", "Macro/function keys",
        ],
        "hazards": [
            "Wireless models may contain batteries.",
            "Dust and debris can accumulate between keys.",
            "Damaged cables may expose electrical conductors.",
            "Burning plastic and circuit boards can release harmful fumes.",
            "Backlight LEDs may contain trace hazardous substances.",
        ],
        "safety": [
            "Disconnect the keyboard before cleaning.",
            "Remove user-replaceable batteries when appropriate.",
            "Keep liquids away from internal electronics.",
            "Do not burn the keyboard or circuit board.",
            "Use anti-static precautions when handling PCBs.",
        ],
        "reusable": [
            "Functional keyboard", "Compatible keycaps and switches",
            "Working USB cable", "Compatible wireless receiver",
            "Tested controller board", "Reusable stabilizers and mounting plates",
        ],
        "repair": [
            "Clean dust from keys and the enclosure.",
            "Inspect the cable and USB connector.",
            "Replace compatible damaged keycaps or switches.",
            "Check batteries and wireless connectivity.",
            "Reflow or repair broken PCB traces where practical.",
            "Replace worn silicone domes or springs.",
        ],
        "recommendation": [
            "Reuse or donate working keyboards.",
            "Repair minor faults when practical.",
            "Send unusable keyboards to an authorized e-waste recycler.",
            "Prefer repairable mechanical keyboards for longevity.",
        ],
        "disposal": [
            "Do not burn keyboard components.",
            "Separate removable batteries for appropriate collection.",
            "Use authorized e-waste collection facilities.",
            "Remove personal data from programmable keyboards.",
        ],
        "recyclingProcess": [
            "Manual dismantling and sorting",
            "Plastic separation and regrinding",
            "PCB sent to specialist smelters",
            "Metal recovery (copper, steel, aluminium)",
            "Cable recycling for copper recovery",
        ],
        "recyclingValue": {
            "Copper": "Moderate value",
            "ABS Plastic": "Low–moderate value",
            "PCB (gold/silver traces)": "Moderate–high value",
            "Steel": "Low value",
        },
        "recoveryRate": "70–85% of total material mass",
        "environmentalImpact": [
            "Plastic pollution if landfilled",
            "Toxic fumes if burned",
            "E-waste accumulation in landfills",
        ],
        "co2Savings": "Recycling saves ~50% CO₂ vs. virgin plastic/metal production",
        "certifications": ["CE", "FCC", "RoHS", "REACH"],
        "regulations": ["EU WEEE Directive", "US EPA e-waste guidelines"],
        "lifespan": "3–10 years (membrane: 3–5; mechanical: 7–10+)",
        "energyRating": "Wired: ~0.5–2W; Wireless: ~1–5W",
        "commonFaults": [
            "Stuck or unresponsive keys", "Cable breakage",
            "Membrane wear", "Liquid damage", "Wireless pairing failure",
        ],
        "didYouKnow": "The QWERTY layout was designed in the 1870s to reduce typewriter jams — and still dominates 150+ years later.",
    },

    "Microwave": {
        "category": "Home Appliance / Kitchen Equipment",
        "alsoKnownAs": ["Microwave Oven", "Microwave Cooker"],
        "description": "An electrical kitchen appliance that uses microwave energy to heat and cook food. It contains a high-voltage power system, electronic controls, a cooking chamber and safety interlocks. Magnetrons generate 2.45 GHz electromagnetic waves that excite water molecules.",
        "uses": [
            "Heating prepared food", "Cooking suitable food items",
            "Defrosting frozen food", "Reheating beverages",
            "Supporting domestic and commercial food preparation",
            "Sterilizing small non-metallic items in lab settings",
        ],
        "materials": [
            "Steel and other metals",
            "Plastic control-panel and enclosure parts",
            "Glass turntable and viewing window materials",
            "Copper wiring and transformer windings",
            "Electronic circuit board materials",
            "Ceramic insulation and other specialized materials",
            "Mica waveguide cover", "Aluminium mesh for door shielding",
        ],
        "composition": {
            "Steel": "50–60%", "Plastics": "10–15%", "Glass": "5–10%",
            "Copper": "5–10%", "Electronics": "5–10%", "Other": "5–10%",
        },
        "components": [
            "Magnetron", "High-voltage transformer or inverter",
            "High-voltage capacitor", "High-voltage diode", "Waveguide",
            "Cooking chamber", "Door and safety interlock switches",
            "Control panel and PCB", "Turntable motor", "Glass turntable",
            "Cooling fan", "Power cord", "Thermal protection devices",
            "Mica waveguide cover", "Door seal and choke",
        ],
        "hazards": [
            "High-voltage components can retain dangerous electrical charge even after unplugging.",
            "The magnetron and high-voltage circuitry can cause severe injury.",
            "Damaged door seals or interlocks may compromise safety.",
            "Broken glass can cause cuts.",
            "Improper dismantling can damage components and expose hazardous materials.",
            "Microwave leakage from damaged doors can cause burns.",
            "Capacitors may hold 2,000+ volts after power-off.",
        ],
        "safety": [
            "Unplug the appliance before external cleaning or inspection.",
            "Never dismantle the high-voltage section yourself.",
            "Do not attempt to discharge the high-voltage capacitor without specialist training.",
            "Do not use an appliance with a damaged door or faulty interlocks.",
            "Have internal electrical repairs performed by a qualified technician.",
            "Never operate an empty microwave for extended periods.",
            "Avoid placing metal objects inside the cooking chamber.",
        ],
        "reusable": [
            "Functional microwave after safety testing",
            "Glass turntable after inspection", "Turntable roller ring",
            "Exterior housing and suitable metal parts",
            "Tested motors and control components recovered professionally",
            "Magnetron if professionally tested",
        ],
        "repair": [
            "Check the power supply and external power cord.",
            "Inspect the door, hinges and latch for visible damage.",
            "Have heating, control-panel and turntable faults diagnosed professionally.",
            "Use compatible replacement parts.",
            "Test safety interlocks and appliance operation after repair.",
            "Replace mica waveguide cover if burnt or soiled.",
        ],
        "recommendation": [
            "Repair and reuse the appliance if it is safe and economical.",
            "Donate a functional appliance after appropriate inspection.",
            "Send irreparable units to an authorized appliance or e-waste recycler.",
            "Allow qualified personnel to handle high-voltage assemblies.",
        ],
        "disposal": [
            "Do not dismantle the high-voltage system at home.",
            "Do not burn cables or plastic components.",
            "Transport the appliance safely to an authorized collection facility.",
            "Ensure electronic components and recoverable metals are processed appropriately.",
            "Discharge capacitors professionally before recycling.",
        ],
        "recyclingProcess": [
            "Manual removal of hazardous components (capacitor, magnetron)",
            "Shredding and magnetic separation of steel",
            "Non-ferrous metal recovery (copper, aluminium)",
            "Plastic separation and recycling",
            "PCB sent to specialist smelters",
            "Glass turntable recycling",
        ],
        "recyclingValue": {
            "Steel": "Moderate value", "Copper": "High value",
            "Aluminium": "Moderate value", "PCB": "Moderate–high value",
            "Magnetron": "Specialist recovery",
        },
        "recoveryRate": "80–90% of total material mass",
        "environmentalImpact": [
            "Scrap metal waste if landfilled",
            "Hazardous capacitor chemicals",
            "Ozone-depleting substances in older units",
            "E-waste accumulation",
        ],
        "co2Savings": "Recycling saves ~60% CO₂ vs. virgin steel/copper production",
        "certifications": ["CE", "UL", "FCC", "RoHS", "FDA (food safety)"],
        "regulations": ["EU WEEE Directive", "EU Ecodesign", "US DOE efficiency standards"],
        "lifespan": "7–10 years",
        "energyRating": "700–1500W typical",
        "commonFaults": [
            "Magnetron failure", "Door interlock failure",
            "Turntable motor failure", "Capacitor failure",
            "Control panel malfunction",
        ],
        "didYouKnow": "Microwave ovens were accidentally invented in 1945 when Percy Spencer noticed a chocolate bar melted near a magnetron.",
    },

    "Mobile": {
        "category": "Consumer Electronics / Communication Device",
        "alsoKnownAs": ["Smartphone", "Cell Phone", "Handset", "Mobile Phone"],
        "description": "A portable electronic communication device that integrates computing, wireless communication, a touchscreen, cameras, sensors, storage and a rechargeable battery. Modern smartphones contain more computing power than Apollo 11's guidance computer.",
        "uses": [
            "Making calls and sending messages",
            "Internet browsing and online learning",
            "Photography and video recording",
            "Navigation and location-based applications",
            "Digital payments and mobile applications",
            "Entertainment and communication",
            "Health monitoring and fitness tracking",
            "Mobile banking and e-commerce",
        ],
        "materials": [
            "Glass and plastic", "Aluminium and other structural metals",
            "Copper wiring and circuit-board tracks",
            "Silicon semiconductor materials",
            "Lithium-ion battery materials",
            "Small quantities of precious and specialty metals",
            "Rare earth elements (neodymium, dysprosium)",
            "Tantalum capacitors", "Indium tin oxide (touchscreen)",
        ],
        "composition": {
            "Glass": "20–30%", "Plastics": "15–20%",
            "Metals (Al, Cu, steel)": "20–30%", "Battery": "10–15%",
            "PCB/Electronics": "10–15%", "Precious metals": "<1%",
        },
        "components": [
            "Display screen", "Touchscreen digitizer", "Motherboard",
            "Processor", "RAM and internal storage",
            "Front and rear camera modules", "Rechargeable battery",
            "Charging port", "Speakers", "Microphone", "SIM tray",
            "Antenna", "Sensors", "Vibration motor",
            "Power and volume buttons", "NFC coil", "Wireless charging coil",
        ],
        "hazards": [
            "Damaged lithium-ion batteries can catch fire.",
            "Broken glass can cause cuts.",
            "Personal information may remain in internal storage.",
            "Improper recycling may release pollutants.",
            "Damaged charging components may present electrical risks.",
            "Rare earth mining causes environmental damage.",
            "Conflict minerals may be present in components.",
        ],
        "safety": [
            "Back up important information before transfer.",
            "Sign out of accounts and securely erase personal data.",
            "Remove SIM and removable memory cards.",
            "Do not puncture or forcibly remove a glued-in battery.",
            "Seek professional assistance for damaged batteries or screens.",
            "Use certified chargers to prevent overheating.",
            "Enable device encryption before disposal.",
        ],
        "reusable": [
            "Working mobile phone after data removal",
            "Compatible screen and camera modules",
            "Tested speakers and charging assemblies",
            "Compatible charger and accessories",
            "Serviceable electronic components",
            "Reusable batteries with acceptable health",
        ],
        "repair": [
            "Replace damaged screens or compatible batteries.",
            "Check charging-port and connector faults.",
            "Diagnose software, storage and connectivity problems.",
            "Test cameras, speakers, microphones and sensors.",
            "Use suitable replacement parts and qualified repair services.",
            "Reflash firmware or replace failed ICs where practical.",
        ],
        "recommendation": [
            "Reuse, repair, trade in or donate a functional phone.",
            "Protect personal information before transferring ownership.",
            "Use manufacturer take-back programs where available.",
            "Send unusable phones to authorized e-waste recyclers.",
            "Support right-to-repair legislation.",
        ],
        "disposal": [
            "Do not put mobile phones in ordinary household waste.",
            "Do not burn or break the device.",
            "Ensure battery-safe handling during collection.",
            "Use an authorized collection or recycling facility.",
            "Remove or wipe all personal data before disposal.",
        ],
        "recyclingProcess": [
            "Data sanitization and testing",
            "Manual dismantling (screen, battery, PCB)",
            "Battery sent to specialist recyclers",
            "Precious metal recovery (gold, silver, palladium)",
            "Copper and aluminium recovery",
            "Rare earth element recovery (emerging tech)",
            "Plastic recycling",
        ],
        "recyclingValue": {
            "Gold": "~0.03g per phone (high value)",
            "Silver": "~0.34g per phone", "Copper": "~16g per phone",
            "Palladium": "Trace amounts", "Rare earths": "Small but strategic value",
        },
        "recoveryRate": "80–95% of material mass with advanced processing",
        "environmentalImpact": [
            "Toxic heavy metals (lead, mercury, cadmium) in soil",
            "Groundwater contamination",
            "E-waste is fastest-growing waste stream globally",
            "Rare earth mining causes habitat destruction",
        ],
        "co2Savings": "Recycling one phone saves ~2 kg CO₂ vs. new production",
        "certifications": ["CE", "FCC", "RoHS", "REACH", "SAR", "IP68"],
        "regulations": [
            "EU WEEE Directive", "EU Right to Repair",
            "US EPA e-waste rules", "India E-Waste Management Rules 2016",
        ],
        "lifespan": "2–5 years typical",
        "energyRating": "3.7–4.5V battery; 5–20W charging",
        "commonFaults": [
            "Battery degradation", "Screen damage", "Charging port failure",
            "Water damage", "Software obsolescence",
        ],
        "didYouKnow": "There are more mobile phones on Earth than people — over 8 billion devices.",
    },

    "Mouse": {
        "category": "Computer Peripheral / Pointing Device",
        "alsoKnownAs": ["Pointing Device", "Computer Mouse", "Cursor Controller"],
        "description": "A computer pointing device that converts hand movements and button actions into cursor movement and commands. It may be wired or wireless and commonly uses optical sensing. Modern mice may have 20+ buttons and 25,000+ DPI sensors.",
        "uses": [
            "Moving the cursor", "Selecting and opening files",
            "Clicking and scrolling", "Computer gaming",
            "Graphic design and navigation", "3D modeling and CAD work",
            "Accessibility input",
        ],
        "materials": [
            "ABS plastic and other polymers", "Copper wiring",
            "Printed circuit board materials", "Steel and other metals",
            "Rubber grips and feet", "Battery materials in wireless models",
            "Gold-plated contacts in premium models",
            "PTFE feet for smooth gliding",
        ],
        "composition": {
            "Plastics": "60–70%", "Metals": "10–15%",
            "PCB/Electronics": "10–15%", "Rubber": "5–10%",
            "Battery (wireless)": "2–5%",
        },
        "components": [
            "Left and right buttons", "Scroll wheel", "Optical or laser sensor",
            "Sensor lens", "Printed circuit board", "Controller chip",
            "Internal switches", "USB cable or wireless receiver",
            "Battery in wireless models", "Outer housing", "Side buttons",
            "DPI adjustment buttons", "RGB LEDs in gaming models",
        ],
        "hazards": [
            "Wireless models may contain batteries.",
            "Damaged cables may expose conductors.",
            "Circuit boards require appropriate recycling.",
            "Burning plastic can release harmful fumes.",
            "Button switches may contain trace heavy metals.",
        ],
        "safety": [
            "Disconnect the mouse before cleaning.",
            "Remove user-replaceable batteries when appropriate.",
            "Avoid liquid entering the electronics.",
            "Do not burn or crush the device.",
            "Use anti-static precautions when handling PCBs.",
        ],
        "reusable": [
            "Working wired or wireless mouse", "Compatible wireless receiver",
            "Undamaged cable", "Working buttons and scroll wheel",
            "Tested optical sensor", "Reusable PTFE feet and glides",
        ],
        "repair": [
            "Clean the optical sensor and lens.",
            "Inspect the cable and connector.",
            "Replace batteries or recharge the device.",
            "Check faulty buttons and scrolling mechanisms.",
            "Test cursor movement and wireless connectivity.",
            "Replace worn switches (Omron, Kailh, etc.).",
            "Replace worn PTFE feet.",
        ],
        "recommendation": [
            "Reuse or donate a working mouse.",
            "Repair minor faults using compatible parts.",
            "Send unusable mice to authorized e-waste collection.",
            "Prefer repairable models with replaceable switches.",
        ],
        "disposal": [
            "Do not burn electronic components.",
            "Separate removable batteries when appropriate.",
            "Use authorized e-waste collection facilities.",
            "Remove personal data from programmable gaming mice.",
        ],
        "recyclingProcess": [
            "Manual dismantling",
            "Plastic separation and regrinding",
            "PCB sent to specialist smelters",
            "Copper recovery from cables",
            "Battery removal for separate processing",
        ],
        "recyclingValue": {
            "Copper": "Moderate value", "ABS Plastic": "Low value",
            "PCB": "Moderate value", "Gold (contacts)": "Trace but valuable",
        },
        "recoveryRate": "70–85% of material mass",
        "environmentalImpact": [
            "Plastic pollution", "Battery hazards if landfilled",
            "E-waste accumulation",
        ],
        "co2Savings": "Recycling saves ~50% CO₂ vs. virgin production",
        "certifications": ["CE", "FCC", "RoHS", "REACH"],
        "regulations": ["EU WEEE Directive", "US EPA e-waste guidelines"],
        "lifespan": "2–7 years depending on quality",
        "energyRating": "Wired: ~0.5–1W; Wireless: ~1–3W",
        "commonFaults": [
            "Double-click issue (switch wear)", "Scroll wheel failure",
            "Cable breakage", "Sensor malfunction", "Wireless connectivity loss",
        ],
        "didYouKnow": "The first computer mouse was invented by Douglas Engelbart in 1964 — it was made of wood.",
    },

    "PCB": {
        "category": "Electronic Component / Circuit Board",
        "alsoKnownAs": ["Printed Circuit Board", "Circuit Board", "PCBA", "Motherboard (contextual)"],
        "description": "A printed circuit board is an insulating board with conductive tracks that mechanically supports and electrically connects electronic components in computers, appliances and other devices. PCBs are the backbone of virtually all modern electronics.",
        "uses": [
            "Connecting electronic components",
            "Processing and controlling electrical signals",
            "Managing power distribution", "Supporting communication systems",
            "Controlling appliances and computer hardware",
            "Embedded systems and IoT devices",
            "Automotive and aerospace electronics",
            "Medical device circuitry",
        ],
        "materials": [
            "Fiberglass-reinforced epoxy laminate", "Copper conductive tracks",
            "Solder alloys", "Silicon semiconductor components",
            "Small quantities of precious and specialty metals",
            "Protective coatings and insulating materials",
            "Gold, silver, palladium in contacts",
            "Tantalum in capacitors", "Lead in older solder (pre-RoHS)",
        ],
        "composition": {
            "Substrate (FR-4)": "40–60%", "Copper": "10–20%",
            "Solder": "5–15%", "Precious metals": "<1%",
            "Components": "10–25%", "Plastics/coatings": "5–10%",
        },
        "components": [
            "Integrated circuits", "Resistors", "Capacitors", "Diodes",
            "Transistors", "Inductors", "Connectors", "Copper tracks",
            "Solder joints", "Oscillators and crystals", "PCB substrate",
            "Power-management components", "Microcontrollers", "Memory chips",
        ],
        "hazards": [
            "Some boards contain lead or other substances requiring controlled handling.",
            "Cut edges can be sharp.",
            "Burning boards releases hazardous smoke and pollutants.",
            "Grinding may produce hazardous dust.",
            "Improper chemical extraction can contaminate the environment.",
            "Capacitors may retain charge.",
            "Beryllium oxide in some components is toxic.",
        ],
        "safety": [
            "Do not burn, grind or sand circuit boards.",
            "Avoid household chemicals for metal recovery.",
            "Handle damaged boards carefully.",
            "Have electrically charged components assessed by qualified personnel.",
            "Use appropriate professional recycling procedures.",
            "Use anti-static wrist straps when handling.",
            "Discharge capacitors safely before handling.",
        ],
        "reusable": [
            "Tested circuit boards",
            "Compatible integrated circuits and connectors",
            "Functional electronic components",
            "Copper and other materials recovered by specialist recyclers",
            "Rare earth magnets from components",
            "Reusable connectors and sockets",
        ],
        "repair": [
            "Inspect for damaged tracks and solder joints.",
            "Diagnose faulty components using appropriate tools.",
            "Replace compatible defective components where practical.",
            "Test the board after repair.",
            "Use qualified electronics technicians for complex faults.",
            "Reflow solder joints and repair traces.",
            "Replace bulging capacitors.",
        ],
        "recommendation": [
            "Reuse functional boards after testing.",
            "Repair boards when technically and economically feasible.",
            "Send unusable PCBs to specialist authorized recyclers.",
            "Use controlled material recovery rather than informal processing.",
            "Support extended producer responsibility.",
        ],
        "disposal": [
            "Never burn circuit boards.",
            "Do not extract metals using household acids.",
            "Keep PCBs separate from ordinary waste where collection is available.",
            "Send them to appropriate electronic-waste recycling facilities.",
            "Remove batteries and capacitors before recycling.",
        ],
        "recyclingProcess": [
            "Manual dismantling and component removal",
            "Shredding and density separation",
            "Pyrometallurgical smelting (copper, precious metals)",
            "Hydrometallurgical leaching (gold, silver, palladium)",
            "Electrorefining for high-purity metals",
            "Plastic/fiberglass substrate recycling",
        ],
        "recyclingValue": {
            "Gold": "~0.2–1g per kg PCB (high value)",
            "Silver": "~1–5g per kg PCB", "Copper": "~100–300g per kg PCB",
            "Palladium": "Trace but valuable", "Tin": "Moderate value",
        },
        "recoveryRate": "90–98% of metal content with advanced processing",
        "environmentalImpact": [
            "Toxic heavy metals (lead, mercury, cadmium)",
            "Brominated flame retardants",
            "Soil and water contamination from informal recycling",
            "Air pollution from open burning",
        ],
        "co2Savings": "Recycling PCBs saves ~80% CO₂ vs. virgin metal mining",
        "certifications": ["UL 94 (flame retardancy)", "IPC-A-610", "RoHS", "REACH"],
        "regulations": [
            "EU WEEE Directive", "Basel Convention",
            "US EPA e-waste rules", "China RoHS",
        ],
        "lifespan": "5–20 years depending on application",
        "energyRating": "Varies: µW to kW depending on board type",
        "commonFaults": [
            "Cold solder joints", "Bulging capacitors", "Broken traces",
            "Component failure", "Corrosion from moisture",
        ],
        "didYouKnow": "A single smartphone PCB can contain over 60 chemical elements — more than the periodic table's stable elements.",
    },

    "Player": {
        "category": "Consumer Electronics / Media Device",
        "alsoKnownAs": ["Media Player", "CD Player", "DVD Player", "MP3 Player", "Blu-ray Player"],
        "description": "An electronic media player, such as a CD player, DVD player or digital audio player, that reads or processes stored media and produces audio or video output. Modern players support streaming and network connectivity.",
        "uses": [
            "Playing music and recorded audio",
            "Playing CDs, DVDs or other compatible media",
            "Playing digital audio or video files",
            "Connecting to televisions, speakers or amplifiers",
            "Supporting home entertainment systems",
            "Streaming online content (smart models)",
            "Karaoke and presentation playback",
        ],
        "materials": [
            "Plastic enclosure materials",
            "Steel and aluminium structural parts",
            "Copper wiring and connectors",
            "Printed circuit board materials",
            "Silicon semiconductor components",
            "Magnets and motor materials",
            "Battery materials in portable models",
            "Glass lens in optical pickup",
            "Gold-plated connectors in premium models",
        ],
        "composition": {
            "Plastics": "40–50%", "Metals": "25–35%",
            "PCB/Electronics": "15–20%", "Optical drive": "5–10%",
            "Battery (portable)": "2–5%",
        },
        "components": [
            "Main printed circuit board",
            "Processor or media decoder",
            "Optical disc drive and laser pickup in CD/DVD models",
            "Spindle motor", "Disc tray and loading mechanism",
            "Audio output circuitry", "Video output circuitry where applicable",
            "USB port or memory-card reader where supported",
            "Display and control buttons", "Power supply",
            "Audio and video connectors", "Battery in portable models",
            "Remote control receiver", "Network module in smart models",
        ],
        "hazards": [
            "Internal power supplies may retain electrical charge.",
            "Optical pickup assemblies require appropriate handling.",
            "Damaged power cables may expose conductors.",
            "Portable models may contain batteries.",
            "Electronic components require suitable recycling.",
            "Laser diodes in optical pickups can damage eyes.",
            "Heavy transformers may cause lifting injuries.",
        ],
        "safety": [
            "Disconnect the player before external inspection.",
            "Do not open power-supply assemblies without proper training.",
            "Do not look directly into an exposed optical pickup.",
            "Remove user-replaceable batteries when appropriate.",
            "Avoid burning or crushing electronic components.",
            "Discharge capacitors before handling.",
            "Use anti-static precautions when handling PCBs.",
        ],
        "reusable": [
            "Working media player", "Compatible remote control",
            "Undamaged power cable and audio/video cables",
            "Functional buttons and display",
            "Tested drive or control-board components",
            "Reusable optical pickup if tested",
            "Reusable motors and mechanisms",
        ],
        "repair": [
            "Check the power supply and external connections.",
            "Clean accessible disc surfaces and inspect loading mechanisms.",
            "Test supported audio and video outputs.",
            "Check remote-control batteries and receiver operation.",
            "Have internal optical-drive or power faults repaired professionally.",
            "Replace belts and gears in disc mechanisms.",
            "Clean or replace laser pickup lens.",
        ],
        "recommendation": [
            "Reuse or donate functional media players.",
            "Repair devices when suitable parts are available.",
            "Recover compatible accessories after inspection.",
            "Send irreparable players to an authorized e-waste recycler.",
            "Consider repurposing as digital media centers.",
        ],
        "disposal": [
            "Do not burn the device or its cables.",
            "Separate removable batteries where appropriate.",
            "Use authorized electronic-waste collection facilities.",
            "Follow safe handling procedures for power-supply assemblies.",
            "Remove personal data from network-connected players.",
        ],
        "recyclingProcess": [
            "Manual dismantling",
            "Optical drive removal for specialist processing",
            "PCB sent to specialist smelters",
            "Copper and aluminium recovery",
            "Plastic separation and recycling",
            "Battery removal for separate processing",
        ],
        "recyclingValue": {
            "Copper": "Moderate–high value",
            "Aluminium": "Moderate value",
            "PCB": "Moderate–high value",
            "Steel": "Low–moderate value",
        },
        "recoveryRate": "75–85% of total material mass",
        "environmentalImpact": [
            "E-waste accumulation in landfills",
            "Toxic fumes if burned",
            "Heavy metal leaching from PCBs",
            "Plastic pollution",
        ],
        "co2Savings": "Recycling saves ~55% CO₂ vs. virgin production",
        "certifications": ["CE", "FCC", "RoHS", "REACH", "Energy Star"],
        "regulations": ["EU WEEE Directive", "US EPA e-waste rules"],
        "lifespan": "5–10 years",
        "energyRating": "10–100W depending on model",
        "commonFaults": [
            "Laser pickup failure", "Disc loading mechanism jam",
            "Spindle motor wear", "Power supply failure",
            "Tray belt breakage",
        ],
        "didYouKnow": "The CD format was co-developed by Philips and Sony in 1982 — a standard CD holds 74 minutes of audio, reportedly to fit Beethoven's 9th Symphony.",
    },

    "Printer": {
        "category": "Computer Peripheral / Output Device",
        "alsoKnownAs": ["Inkjet Printer", "Laser Printer", "All-in-One", "MFP"],
        "description": "An electronic output device that produces text, images and graphics on paper or other supported media. Printers may use inkjet, laser, thermal or other printing technologies. Multifunction printers also scan, copy and fax.",
        "uses": [
            "Printing documents and reports",
            "Printing photographs and graphics",
            "Producing labels and receipts",
            "Printing educational and business materials",
            "Scanning and copying in multifunction printer models",
            "3D printing (in additive manufacturing variants)",
        ],
        "materials": [
            "Plastic enclosure materials",
            "Steel and aluminium structural parts",
            "Copper wiring and electronic circuit boards",
            "Rubber rollers and drive belts",
            "Glass in scanner assemblies",
            "Ink, toner and printing consumables",
            "Motors, gears and other mechanical materials",
            "Photoconductor drums (laser models)",
        ],
        "composition": {
            "Plastics": "40–55%", "Metals": "20–30%",
            "PCB/Electronics": "10–15%", "Mechanical parts": "10–15%",
            "Consumables (ink/toner)": "2–5%",
        },
        "components": [
            "Main control board", "Printhead in applicable models",
            "Ink cartridges or toner cartridge", "Paper feed rollers",
            "Paper tray", "Stepper motors and drive mechanisms",
            "Power supply", "USB or network interface",
            "Display and control panel",
            "Scanner assembly in multifunction models",
            "Waste-ink container or toner collection components where applicable",
            "Fuser assembly (laser models)",
        ],
        "hazards": [
            "Toner and ink may irritate skin or eyes.",
            "Laser printers may contain high-voltage components.",
            "Fuser assemblies in laser printers can remain hot after use.",
            "Broken scanner glass may cause cuts.",
            "Improper handling can release toner dust.",
            "Some printers store network credentials or documents.",
        ],
        "safety": [
            "Disconnect the printer before cleaning or inspection.",
            "Allow hot components to cool before handling.",
            "Follow manufacturer instructions for toner and ink replacement.",
            "Avoid inhaling toner dust or blowing it into the air.",
            "Do not dismantle high-voltage or laser assemblies without training.",
            "Remove personal information from network-connected devices before transfer.",
        ],
        "reusable": [
            "Functional printer",
            "Compatible paper trays and external cables",
            "Working scanner assembly",
            "Tested motors, rollers and control components",
            "Compatible reusable accessories",
        ],
        "repair": [
            "Check power and USB or network connections.",
            "Inspect paper jams and clean accessible rollers.",
            "Replace compatible cartridges and consumables.",
            "Diagnose printhead, scanner and paper-feed faults.",
            "Update supported printer drivers and firmware.",
            "Use qualified technicians for internal electrical repairs.",
        ],
        "recommendation": [
            "Reuse or donate working printers.",
            "Repair paper-feed or other economically repairable faults.",
            "Follow manufacturer programs for cartridge returns.",
            "Send unusable printers to an authorized e-waste recycler.",
        ],
        "disposal": [
            "Do not burn the printer or its cartridges.",
            "Handle toner and ink according to manufacturer instructions.",
            "Use separate cartridge collection where available.",
            "Send the electronic device to an authorized recycler.",
        ],
        "recyclingProcess": [
            "Toner/ink cartridge removal for specialist processing",
            "Manual dismantling of mechanical assemblies",
            "Plastic separation and regrinding",
            "PCB sent to specialist smelters",
            "Metal recovery (steel, copper, aluminium)",
            "Scanner glass recycling",
        ],
        "recyclingValue": {
            "Copper": "Moderate value", "Steel": "Moderate value",
            "Aluminium": "Moderate value", "PCB": "Moderate–high value",
            "Toner (reusable)": "Moderate value",
        },
        "recoveryRate": "75–85% of total material mass",
        "environmentalImpact": [
            "Toner dust contamination if improperly handled",
            "Plastic pollution",
            "Heavy metals from PCBs",
            "Ink chemicals can leach into soil",
        ],
        "co2Savings": "Recycling saves ~55% CO₂ vs. virgin production",
        "certifications": ["CE", "FCC", "RoHS", "REACH", "Energy Star"],
        "regulations": [
            "EU WEEE Directive", "EU Ecodesign",
            "US EPA e-waste rules", "Energy Star program",
        ],
        "lifespan": "3–7 years typical",
        "energyRating": "Inkjet: 20–100W; Laser: 300–1500W",
        "commonFaults": [
            "Paper jam", "Printhead clog (inkjet)",
            "Toner cartridge issues (laser)", "Fuser failure",
            "Roller wear", "Network connectivity problems",
        ],
        "didYouKnow": "The first computer printer, the IBM 1403, could print 1,100 lines per minute — faster than many modern inkjets.",
    },

    "Television": {
        "category": "Consumer Electronics / Display Device",
        "alsoKnownAs": ["TV", "Smart TV", "LED TV", "OLED TV", "LCD TV", "CRT TV"],
        "description": "An electronic display and entertainment device that receives or displays video and audio signals. Television technologies include LCD, LED-backlit LCD, OLED and older CRT systems. Modern smart TVs integrate internet connectivity and streaming apps.",
        "uses": [
            "Watching television programs and films",
            "Displaying video from connected devices",
            "Streaming digital media on smart TVs",
            "Gaming and multimedia entertainment",
            "Displaying educational and informational content",
            "Digital signage and commercial display",
        ],
        "materials": [
            "Glass and specialized display materials",
            "Plastic or metal enclosure",
            "Copper wiring and circuit-board tracks",
            "Semiconductor and electronic components",
            "Backlight materials in applicable LCD models",
            "Steel and aluminium structural parts",
            "Additional specialized materials depending on display technology",
            "Rare earth phosphors (older models)",
            "Lead in CRT glass",
        ],
        "composition": {
            "Glass/Display panel": "30–45%", "Plastics": "20–30%",
            "Metals": "15–25%", "PCB/Electronics": "5–10%",
            "Backlight materials": "2–5%",
        },
        "components": [
            "Display panel", "Mainboard", "Power supply board",
            "Display driver board", "Backlight assembly in LCD models",
            "Speakers", "Tuner and signal-processing circuitry",
            "HDMI, USB and antenna connectors",
            "Remote-control receiver", "Wireless module in smart TVs",
            "Power cable", "Stand or mounting hardware",
        ],
        "hazards": [
            "Broken screens may cause cuts.",
            "Older CRT televisions can contain leaded glass and other hazardous materials.",
            "Power supplies may retain dangerous electrical charge.",
            "Large televisions can cause injury if dropped.",
            "Improper dismantling and burning may release pollutants.",
            "LCD panels contain liquid crystals that can be toxic.",
        ],
        "safety": [
            "Disconnect the television before external inspection.",
            "Do not break or puncture the screen.",
            "Never open power-supply assemblies without appropriate training.",
            "Use suitable lifting and transport precautions.",
            "Have damaged displays handled by qualified recycling personnel.",
        ],
        "reusable": [
            "Functional television", "Compatible remote control",
            "Undamaged power and signal cables",
            "Working stand or mounting hardware",
            "Tested electronic boards recovered professionally",
        ],
        "repair": [
            "Check the power source and external connections.",
            "Test signal inputs and remote-control functions.",
            "Diagnose sound, display, backlight and connectivity problems.",
            "Use compatible replacement components.",
            "Have internal power and display faults assessed professionally.",
        ],
        "recommendation": [
            "Reuse or donate a functional television.",
            "Repair suitable display, audio or connection faults.",
            "Send damaged televisions to authorized e-waste recyclers.",
            "Use specialist facilities for older CRT technology.",
        ],
        "disposal": [
            "Do not break or burn television components.",
            "Keep damaged displays safely contained.",
            "Use authorized e-waste collection facilities.",
            "Follow specialized handling requirements for legacy CRT televisions.",
        ],
        "recyclingProcess": [
            "Manual dismantling and screen separation",
            "CRT glass processing (lead extraction) where applicable",
            "Plastic separation and recycling",
            "PCB sent to specialist smelters",
            "Metal recovery (copper, aluminium, steel)",
            "Backlight (mercury) safe extraction in older LCDs",
        ],
        "recyclingValue": {
            "Copper": "Moderate–high value", "Aluminium": "Moderate value",
            "Gold (PCBs)": "Moderate value", "Steel": "Low–moderate value",
            "Glass (panel)": "Low value, high volume",
        },
        "recoveryRate": "80–90% of total material mass",
        "environmentalImpact": [
            "Lead leaching from CRT glass",
            "Mercury from older LCD backlights",
            "Brominated flame retardants from plastics",
            "Soil and groundwater contamination",
        ],
        "co2Savings": "Recycling saves ~60% CO₂ vs. virgin production",
        "certifications": ["CE", "FCC", "RoHS", "REACH", "Energy Star"],
        "regulations": [
            "EU WEEE Directive", "EU RoHS",
            "US EPA e-waste rules", "Energy Star program",
        ],
        "lifespan": "5–12 years",
        "energyRating": "50–400W depending on size and technology",
        "commonFaults": [
            "Backlight failure (LCD)", "Dead pixels", "Power board failure",
            "HDMI port damage", "Panel cracks", "Smart OS obsolescence",
        ],
        "didYouKnow": "A modern 55-inch OLED TV has over 8 million self-lit pixels — each one can turn on and off independently.",
    },

    "Washing Machine": {
        "category": "Home Appliance / Laundry Equipment",
        "alsoKnownAs": ["Washer", "Laundry Machine", "Front Loader", "Top Loader"],
        "description": "An electrical household appliance that washes clothes using water, detergent and mechanical movement. Automatic washing machines use electronic controls, motors, pumps and water-management systems.",
        "uses": [
            "Washing clothes and household fabrics",
            "Rinsing detergent from laundry",
            "Spinning clothes to remove excess water",
            "Supporting domestic and commercial laundry operations",
            "Automating washing cycles and water management",
        ],
        "materials": [
            "Steel and stainless steel",
            "Plastic and polymer enclosure parts",
            "Copper wiring and motor windings",
            "Rubber hoses, seals and door gaskets",
            "Aluminium and other motor or structural materials",
            "Electronic circuit-board materials",
            "Glass in front-loading doors",
            "Concrete counterweights",
        ],
        "composition": {
            "Steel/Stainless": "50–60%", "Plastics": "10–15%",
            "Concrete": "5–10%", "Copper": "3–5%",
            "Electronics": "5–10%", "Other": "5–10%",
        },
        "components": [
            "Outer cabinet", "Drum or wash basket", "Inner tub",
            "Electric motor", "Drive belt or direct-drive assembly",
            "Electronic control board", "Control panel and display",
            "Water inlet valve", "Drain pump", "Water hoses",
            "Door lock and safety interlock",
            "Pressure or water-level sensor",
            "Heating element in applicable models",
            "Suspension springs and shock absorbers", "Power cable",
        ],
        "hazards": [
            "Electrical components can present shock risks if handled while energized.",
            "Water leaks may damage electrical components and surrounding property.",
            "Heavy equipment can cause lifting or crushing injuries.",
            "Moving drums and rotating parts can cause injury during operation.",
            "Sharp sheet-metal edges may cause cuts.",
            "Capacitors and other electrical components may retain charge after disconnection.",
        ],
        "safety": [
            "Disconnect the appliance from electrical power before inspection.",
            "Turn off the water supply before servicing hoses or valves.",
            "Do not bypass door locks or other safety interlocks.",
            "Use appropriate lifting assistance when moving the appliance.",
            "Avoid dismantling electrical components without training.",
            "Have internal electrical faults repaired by a qualified technician.",
        ],
        "reusable": [
            "Functional washing machine",
            "Drum and suitable stainless-steel parts",
            "Compatible motor and pump after testing",
            "Undamaged hoses and fittings",
            "Control panel and electronic boards that pass testing",
            "Suitable metal and plastic parts recovered for reuse",
        ],
        "repair": [
            "Inspect water inlet hoses and visible connections.",
            "Check drain filters according to the manufacturer's instructions.",
            "Diagnose drainage, filling, spinning and vibration problems.",
            "Inspect the door seal for visible damage.",
            "Have motor, control-board and heating faults checked professionally.",
            "Test for leaks and safe operation after repair.",
        ],
        "recommendation": [
            "Repair and reuse a safe, functional appliance.",
            "Donate or resell working machines after inspection.",
            "Replace compatible damaged parts when economically practical.",
            "Send irreparable machines to authorized appliance recyclers.",
            "Recover steel, copper and other materials through controlled recycling.",
        ],
        "disposal": [
            "Do not dismantle the machine while connected to electricity or water.",
            "Use suitable lifting and transport procedures.",
            "Do not dump the appliance in open areas or water bodies.",
            "Use authorized e-waste or appliance collection facilities.",
            "Ensure electronic boards, motors and other materials are processed appropriately.",
        ],
        "recyclingProcess": [
            "Manual removal of hazardous components (capacitors, PCB)",
            "Concrete counterweight separation and recycling",
            "Shredding and magnetic separation of steel",
            "Non-ferrous metal recovery (copper, aluminium)",
            "Plastic separation and recycling",
            "Motor and pump specialist recovery",
        ],
        "recyclingValue": {
            "Steel": "High volume, moderate value",
            "Copper": "High value", "Aluminium": "Moderate value",
            "Stainless steel": "Moderate–high value",
            "PCB": "Moderate value",
        },
        "recoveryRate": "85–95% of total material mass",
        "environmentalImpact": [
            "Scrap metal waste if landfilled",
            "Heavy appliance waste in landfills",
            "Refrigerant/oil leaks in some models",
            "Electronic waste contamination",
        ],
        "co2Savings": "Recycling saves ~65% CO₂ vs. virgin steel production",
        "certifications": ["CE", "UL", "RoHS", "REACH", "Energy Star"],
        "regulations": [
            "EU WEEE Directive", "EU Ecodesign",
            "US EPA e-waste rules", "Energy Star program",
        ],
        "lifespan": "8–12 years",
        "energyRating": "500–2500W depending on cycle",
        "commonFaults": [
            "Drain pump failure", "Door seal leak", "Control board failure",
            "Motor wear", "Bearing failure", "Water inlet valve failure",
        ],
        "didYouKnow": "A front-loading washing machine typically uses 40–50% less water and energy than a top-loader.",
    },
}

FIELD_ALIASES = {
    "category": ("category", "type", "classification"),
    "alsoKnownAs": ("alsoKnownAs", "also_known_as", "aliases", "synonyms"),
    "description": ("description", "about", "device_information"),
    "uses": ("uses", "common_uses"),
    "materials": ("materials", "common_materials"),
    "composition": ("composition", "material_composition", "breakdown"),
    "components": ("components", "common_components"),
    "hazards": ("hazards", "hazardous", "hazardous_materials"),
    "safety": ("safety", "safety_guidance", "handling"),
    "reusable": ("reusable", "reusable_parts", "reuse"),
    "repair": ("repair", "repair_guidance"),
    "recommendation": ("recommendation", "recycling_recommendation"),
    "disposal": ("disposal", "disposal_guidance"),
    "recyclingProcess": ("recyclingProcess", "recycling_process", "recycling_steps"),
    "recyclingValue": ("recyclingValue", "recycling_value", "material_value"),
    "recoveryRate": ("recoveryRate", "recovery_rate"),
    "environmentalImpact": ("environmentalImpact", "environmental_impact", "impact"),
    "co2Savings": ("co2Savings", "co2_savings", "carbon_savings"),
    "lifespan": ("lifespan", "life_expectancy", "service_life"),
    "energyRating": ("energyRating", "energy_rating", "power_rating"),
}

CATEGORY_ALIASES = {
    "cell phone": "mobile phone",
    "phone": "mobile phone",
    "smartphone": "mobile phone",
    "mobile": "mobile phone",
    "pcb": "circuit board",
    "motherboard": "circuit board",
    "computer monitor": "monitor",
    "computer mouse": "mouse",
    "power adapter": "charger",
    "adapter": "charger",
    "headphone": "headphones",
    "lithium battery": "battery",
    "battery pack": "battery",
    "tv": "television",
    "washer": "washing machine",
    "laundry machine": "washing machine",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def normalize_name(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").strip().split())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner="Loading classification model...")
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_class_names(path_string: str) -> list[str]:
    data = load_json(Path(path_string))
    if isinstance(data, list):
        names = [str(x) for x in data]
    elif isinstance(data, dict):
        if all(str(k).isdigit() for k in data):
            names = [str(data[k]) for k in sorted(data, key=lambda x: int(x))]
        elif all(str(v).isdigit() for v in data.values()):
            names = [""] * len(data)
            for name, index in data.items():
                names[int(index)] = str(name)
        else:
            raise ValueError("class_names.json dictionary must map indexes to names or names to indexes.")
    else:
        raise ValueError("class_names.json must contain a list or dictionary.")
    if not names or any(not name for name in names):
        raise ValueError("Class names are empty or incomplete.")
    return names


@st.cache_data(show_spinner=False)
def load_custom_info(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def resolve_info(category: str, custom_info: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_name(category)
    candidates = [normalized]
    for suffix in (" class", " waste", " e waste", " electronic waste"):
        if normalized.endswith(suffix):
            candidates.append(normalized[: -len(suffix)].strip())
    candidates.extend(k for k, v in CATEGORY_ALIASES.items() if v == normalized)
    for candidate in list(candidates):
        if candidate in CATEGORY_ALIASES:
            candidates.append(CATEGORY_ALIASES[candidate])

    base: dict[str, Any] = {}
    for key, value in DEFAULT_INFO.items():
        if normalize_name(key) in candidates:
            base = dict(value)
            break

    custom: dict[str, Any] = {}
    for key, value in custom_info.items():
        if normalize_name(str(key)) in candidates and isinstance(value, dict):
            custom = value
            break

    merged = dict(base)
    merged.update({k: v for k, v in custom.items() if v not in (None, "", [], {})})
    return merged


def field_value(info: dict[str, Any], name: str, default: Any) -> Any:
    for alias in FIELD_ALIASES.get(name, (name,)):
        value = info.get(alias)
        if value not in (None, "", [], {}):
            return value
    return default


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------
def esc(value: Any) -> str:
    return escape(str(value))


def html_block(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines())


def show_html(markup: str) -> None:
    st.markdown(html_block(markup), unsafe_allow_html=True)


def pretty_name(label: str) -> str:
    words = str(label).replace("_", " ").replace("-", " ").split()
    return " ".join(w if w.isupper() else w.capitalize() for w in words) or str(label)


def section(title: str, note: str | None = None, icon: str = "◆") -> None:
    show_html(
        f'<h2 class="sec"><span class="sec-ico">{esc(icon)}</span>{esc(title)}</h2>'
    )
    if note:
        show_html(f'<p class="sec-note">{esc(note)}</p>')


def list_html(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        items = [item for item in value if str(item).strip()]
        if not items:
            return "<p>No information has been added yet.</p>"
        return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"
    return f"<p>{esc(value)}</p>"


def dict_html(value: dict) -> str:
    if not isinstance(value, dict) or not value:
        return "<p>No data available.</p>"
    return "<ul>" + "".join(
        f"<li><b>{esc(k)}:</b> {esc(v)}</li>" for k, v in value.items()
    ) + "</ul>"


def panel(title: str | None, value: Any, tone: str = "") -> str:
    heading = f"<h4>{esc(title)}</h4>" if title else ""
    return f'<div class="panel {tone}">{heading}{list_html(value)}</div>'


def panel_dict(title: str | None, value: dict, tone: str = "") -> str:
    heading = f"<h4>{esc(title)}</h4>" if title else ""
    return f'<div class="panel {tone}">{heading}{dict_html(value)}</div>'


def panel_row(*panels: str, columns: int = 2) -> None:
    cls = "panel-row"
    if columns == 3:
        cls += " three"
    elif columns == 4:
        cls += " four"
    show_html(f'<div class="{cls}">{"".join(panels)}</div>')


def score_band(confidence: float) -> tuple[str, str]:
    if confidence >= HIGH_CONFIDENCE:
        return "High", COLOR_HIGH
    if confidence >= MODERATE_CONFIDENCE:
        return "Moderate", COLOR_MODERATE
    return "Low", COLOR_LOW


def predictions_html(ranked: list[tuple[str, float]]) -> str:
    rows = []
    for index, (label, score) in enumerate(ranked[:TOP_K], start=1):
        width = float(np.clip(score, 0.0, 1.0)) * 100
        first = index == 1
        rows.append(
            f'<div class="pred{" first" if first else ""}">'
            f'<span class="pred-rank">#{index}</span>'
            f'<span class="pred-name">{esc(pretty_name(label))}</span>'
            f'<span class="pred-track"><span class="bar{" top" if first else ""}" style="width:{width:.1f}%"></span></span>'
            f'<span class="pred-val">{score * 100:.1f}%</span></div>'
        )
    return "".join(rows)


def result_card_html(result: dict[str, Any]) -> str:
    category, confidence = result["category"], result["confidence"]
    ranked, margin = result["ranked"], result["margin"]
    band, color = score_band(confidence)

    if confidence < MODERATE_CONFIDENCE:
        note = '<div class="note low">Low model score. Try another clear image and verify the result manually.</div>'
    elif margin is not None and margin < CLOSE_CALL_MARGIN:
        runner_up = pretty_name(ranked[1][0])
        note = f'<div class="note warn">Close call: <b>{esc(runner_up)}</b> is only {margin:.1f} points behind. Check both before relying on this result.</div>'
    elif confidence < HIGH_CONFIDENCE:
        note = '<div class="note warn">Moderate score. Check the other predictions before relying on this result.</div>'
    else:
        note = '<div class="note">The model strongly favors this category, but a high score does not guarantee the prediction is correct.</div>'

    margin_pill = f'<span class="pill plain">+{margin:.1f} pts over #2</span>' if margin is not None else ""
    return (
        '<div class="result">'
        '<div class="rank-badge">🏆 Rank #1 match</div>'
        '<div class="result-top"><div>'
        '<div class="result-label">Predicted category</div>'
        f'<div class="cat-name">{esc(pretty_name(category))}</div>'
        f'<span class="pill" style="color:{color}">{band} confidence</span>{margin_pill}'
        '</div>'
        f'<div class="ring" style="--p:{min(confidence, 100):.1f};--ringc:{color}">'
        f'<div class="ring-in"><b>{confidence:.0f}%</b><span>score</span></div></div>'
        '</div>'
        '<div class="result-divider"></div>'
        '<div class="alts-title">Ranked predictions</div>'
        f'{predictions_html(ranked)}'
        f'{note}'
        '</div>'
    )


def empty_state_html(icon: str, title: str, text: str) -> str:
    return f'<div class="empty"><div class="icon">{icon}</div><h3>{esc(title)}</h3><p>{esc(text)}</p></div>'


def spec_grid_html(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    cards = "".join(
        f'<div class="spec"><div class="spec-key">{esc(k)}</div><div class="spec-val">{esc(v)}</div></div>'
        for k, v in items
    )
    return f'<div class="spec-grid">{cards}</div>'


def composition_html(composition: dict) -> str:
    if not isinstance(composition, dict) or not composition:
        return "<p>No composition data available.</p>"
    rows = []
    for label, pct in composition.items():
        nums = [
            float(s)
            for s in str(pct).replace("%", "").replace("–", "-").replace("—", "-").split("-")
            if s.strip().replace(".", "").isdigit()
        ]
        value = max(nums) if nums else 0
        width = min(value, 100)
        rows.append(
            f'<div class="comp">'
            f'<div class="comp-row"><span>{esc(label)}</span><span>{esc(pct)}</span></div>'
            f'<div class="comp-track"><div class="comp-bar" style="width:{width:.1f}%"></div></div>'
            f'</div>'
        )
    return "".join(rows)


def stat_strip_html(items: list[tuple[str, str, str]]) -> str:
    """items: list of (icon, key, value)."""
    if not items:
        return ""
    cards = "".join(
        f'<div class="stat"><div class="stat-ico">{esc(ico)}</div>'
        f'<div><div class="stat-key">{esc(k)}</div><div class="stat-val">{esc(v)}</div></div></div>'
        for ico, k, v in items
    )
    return f'<div class="stat-strip">{cards}</div>'


def build_report(result: dict[str, Any], info: dict[str, Any]) -> str:
    def as_lines(value: Any) -> str:
        if isinstance(value, dict):
            return "\n".join(f"  - {k}: {v}" for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            return "\n".join(f"  - {item}" for item in value)
        return f"  {value}"

    lines = [
        "E-WASTE CLASSIFICATION REPORT",
        "=" * 32,
        f"Rank #1 category : {pretty_name(result['category'])}",
        f"Model score      : {result['confidence']:.1f}%",
        "",
        "Ranked predictions:",
    ]
    lines += [
        f"  #{i} {pretty_name(label)}: {score * 100:.1f}%"
        for i, (label, score) in enumerate(result["ranked"][:TOP_K], 1)
    ]
    for title, key in (
        ("Category", "category"),
        ("Also known as", "alsoKnownAs"),
        ("Description", "description"),
        ("Common uses", "uses"),
        ("Common materials", "materials"),
        ("Composition", "composition"),
        ("Common components", "components"),
        ("Lifespan", "lifespan"),
        ("Energy rating", "energyRating"),
        ("Recovery rate", "recoveryRate"),
        ("CO2 savings", "co2Savings"),
        ("Potential hazards", "hazards"),
        ("Safe handling", "safety"),
        ("Reusable parts", "reusable"),
        ("Repair guidance", "repair"),
        ("Recycling process", "recyclingProcess"),
        ("Recycling value", "recyclingValue"),
        ("Environmental impact", "environmentalImpact"),
        ("Recycling recommendation", "recommendation"),
        ("Disposal", "disposal"),
    ):
        value = field_value(info, key, "")
        if value:
            lines += ["", f"{title}:", as_lines(value)]
    lines += [
        "",
        "Note: AI predictions can be wrong. Verify the item before acting on safety or disposal advice.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if PREPROCESSING == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(array)
    if PREPROCESSING == "rescale":
        return array / 255.0
    return array


def predict_image(
    model: tf.keras.Model,
    class_names: list[str],
    image: Image.Image,
    file_hash: str,
    use_tta: bool,
) -> dict[str, Any]:
    cache_key = f"prediction_{file_hash}_{use_tta}_{PREPROCESSING}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    array = preprocess(image)
    batch = np.stack([array, array[:, ::-1, :]]) if use_tta else np.expand_dims(array, axis=0)
    outputs = np.asarray(model.predict(batch, verbose=0))
    output = outputs.mean(axis=0).reshape(-1)

    if output.size != len(class_names):
        raise ValueError(
            f"Model returns {output.size} scores, but class_names.json contains "
            f"{len(class_names)} classes. Check class_names.json and your trained model."
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("The model returned invalid prediction values.")

    normalized = bool(np.all(output >= 0) and np.isclose(float(output.sum()), 1.0, atol=0.05))

    order = np.argsort(output)[::-1]
    ranked = [(class_names[i], float(output[i])) for i in order]
    margin = (ranked[0][1] - ranked[1][1]) * 100 if len(ranked) > 1 else None

    result = {
        "category": ranked[0][0],
        "confidence": ranked[0][1] * 100.0,
        "ranked": ranked,
        "margin": margin,
        "normalized": normalized,
        "file_hash": file_hash,
    }
    st.session_state[cache_key] = result
    return result


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
def main() -> None:
    show_html(
        """
        <div class="page-head">
          <span class="orb a"></span>
          <span class="orb b"></span>
          <span class="orb c"></span>
          <span class="eyebrow">AI-powered sorting assistant</span>
          <h1 class="head-title">E-Waste Classification Model</h1>
          <p class="head-sub">Upload a photo of an electronic item to identify its category and get safe reuse, repair and recycling guidance.</p>
          <div class="chips">
            <span class="chip">♻️ Reuse &amp; recycle tips</span>
            <span class="chip">⚠️ Hazard awareness</span>
            <span class="chip">🏆 Ranked predictions</span>
            <span class="chip">🌍 Environmental impact</span>
          </div>
        </div>
        """
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH}`")
        st.info("Check that models/ewaste_model.keras exists in your project folder.")
        st.stop()
    if not CLASSES_PATH.exists():
        st.error(f"Class names file not found: `{CLASSES_PATH}`")
        st.info("Check that models/class_names.json exists.")
        st.stop()

    try:
        model = load_model(str(MODEL_PATH))
        class_names = load_class_names(str(CLASSES_PATH))
        custom_info = load_custom_info(str(INFO_PATH))
    except Exception as exc:
        logger.exception("Could not load model or data")
        st.error(f"Could not load the model or its supporting files: {exc}")
        st.stop()

    upload_col, result_col = st.columns([1, 1.1], gap="large")
    image: Image.Image | None = None
    file_hash: str | None = None

    with upload_col:
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Show the whole device in good light and avoid blurry images.",
        )
        show_html(f'<p class="hint">JPG or PNG, up to {MAX_UPLOAD_MB} MB. Show the whole item in good light.</p>')

        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Image is too large. Please upload an image under {MAX_UPLOAD_MB} MB.")
            else:
                try:
                    raw_bytes = uploaded.getvalue()
                    file_hash = hashlib.sha256(raw_bytes).hexdigest()
                    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                    st.image(image)
                except (UnidentifiedImageError, OSError, ValueError):
                    image, file_hash = None, None
                    st.error("This image could not be opened. Please upload a JPG or PNG image.")

        use_tta = st.toggle(
            "Higher accuracy mode",
            value=True,
            help="Checks the image and its mirror image, then averages the scores. Slightly slower, usually more reliable.",
        )
        run_button = st.button("✦  Classify image", type="primary", disabled=image is None)

        if run_button and image is not None and file_hash is not None:
            try:
                with st.spinner("Classifying image..."):
                    st.session_state["eco_result"] = predict_image(
                        model, class_names, image, file_hash, use_tta
                    )
            except Exception as exc:
                logger.exception("Prediction failed")
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("eco_result")
    has_result = result is not None and image is not None and result.get("file_hash") == file_hash

    with result_col:
        if has_result:
            show_html(result_card_html(result))
            if not result["normalized"]:
                st.warning(
                    "Model outputs do not look like normalized probabilities. "
                    "Displayed percentages may not represent calibrated confidence."
                )
        elif image is None:
            show_html(empty_state_html(
                "📷", "Your result will appear here",
                "Upload a clear photo of an electronic item, then select Classify image.",
            ))
        else:
            show_html(empty_state_html(
                "🔍", "Ready to classify",
                "Select Classify image to see the rank #1 category, its score and the next best matches.",
            ))

    if not has_result:
        show_html(
            '<div class="disclaimer">This tool classifies visible images only and gives general guidance. '
            'Always verify the item before acting on safety or disposal advice.</div>'
        )
        st.stop()

    category = result["category"]
    info = resolve_info(category, custom_info)

    with result_col:
        st.write("")
        st.download_button(
            "📄 Download report (.txt)",
            data=build_report(result, info),
            file_name=f"ewaste_report_{normalize_name(category).replace(' ', '_')}.txt",
            mime="text/plain",
        )

    # --- Quick specs strip -----------------------------------------
    stat_items = []
    for icon, key, label in (
        ("⏳", "lifespan", "Typical lifespan"),
        ("⚡", "energyRating", "Energy rating"),
        ("♻️", "recoveryRate", "Recovery rate"),
        ("🌱", "co2Savings", "CO₂ savings"),
    ):
        value = field_value(info, key, "")
        if value:
            stat_items.append((icon, label, value))
    if stat_items:
        section("Quick specifications", icon="📊")
        show_html(stat_strip_html(stat_items))

    # --- About ------------------------------------------------------
    section("About this device", icon="📖")
    panel_row(
        panel(
            "💡 What it is",
            field_value(
                info, "description",
                f"The model classified the uploaded image as {pretty_name(category)}. "
                "Check the physical item label to confirm its identity.",
            ),
        ),
        panel(
            "🎯 Common uses",
            field_value(info, "uses", "Use and purpose information has not been added for this category yet."),
        ),
    )

    # --- Also known as ---------------------------------------------
    aka = field_value(info, "alsoKnownAs", None)
    category_label = field_value(info, "category", None)
    if aka or category_label:
        chips = []
        if category_label:
            chips.append(f'<span class="chip">🏷️ {esc(category_label)}</span>')
        for name in (aka if isinstance(aka, (list, tuple)) else [aka] if aka else []):
            chips.append(f'<span class="chip">{esc(name)}</span>')
        show_html(f'<div class="chips" style="margin-top:14px">{"".join(chips)}</div>')

    # --- Materials, composition & components -----------------------
    section("Materials and components", "Typical for this category. Actual materials depend on the exact model.", icon="🧩")
    panel_row(
        panel(
            "🧱 Common materials",
            field_value(info, "materials", "Material information is not available for this category yet."),
        ),
        panel(
            "🔩 Common components",
            field_value(info, "components", "Component information is not available for this category yet."),
        ),
    )

    composition = field_value(info, "composition", None)
    if composition:
        st.write("")
        show_html(
            f'<div class="panel"><h4>🧪 Material composition (approximate)</h4>'
            f'{composition_html(composition)}</div>'
        )

    # --- Hazards and safe handling ---------------------------------
    section("Hazards and safe handling", "Do not dismantle unknown electronics based only on an AI prediction.", icon="🛡️")
    hazards = field_value(
        info, "hazards",
        "No category-specific hazard details are in the database. Treat unknown electronics cautiously.",
    )
    panel_row(
        panel("⚠️ Potential hazards", hazards, "danger"),
        panel(
            "🦺 Safe handling",
            field_value(
                info, "safety",
                "Keep dry, avoid burning or dismantling, and consult a qualified e-waste handler.",
            ),
            "safe",
        ),
    )
    if "batter" in str(hazards).lower() or "lithium" in str(hazards).lower():
        st.write("")
        show_html(
            '<div class="note low">🔋 <b>Battery safety:</b> never puncture, crush or burn batteries. '
            'If a battery is swollen, leaking, unusually hot or damaged, do not handle it unnecessarily '
            'and seek professional guidance.</div>'
        )

    # --- Reuse, repair, recycling ----------------------------------
    section("Reuse, repair and recycling", icon="♻️")
    reuse_tab, repair_tab, recycle_tab, value_tab, impact_tab, disposal_tab = st.tabs(
        ["♻️ Reuse", "🔧 Repair", "🏭 Recycle", "💰 Value", "🌍 Impact", "🗑️ Disposal"]
    )
    with reuse_tab:
        show_html(panel(
            "Reusable parts or options",
            field_value(info, "reusable", "Reuse information is not available for this category yet. Ask a technician to assess the item."),
            "plain",
        ))
    with repair_tab:
        show_html(panel(
            "Repair guidance",
            field_value(info, "repair", "Consult a qualified repair technician and check warranty status before opening the device."),
            "plain",
        ))
    with recycle_tab:
        panel_row(
            panel(
                "Recycling recommendation",
                field_value(info, "recommendation", "Use an authorized e-waste collection or recycling facility. Confirm it accepts this device type."),
                "plain",
            ),
            panel(
                "Recycling process",
                field_value(info, "recyclingProcess", "Detailed process data is not available for this category yet."),
                "plain",
            ),
        )
    with value_tab:
        rv = field_value(info, "recyclingValue", None)
        if rv:
            show_html(panel_dict("💰 Recoverable material value", rv, "eco"))
        else:
            show_html(panel("💰 Recoverable material value", "Value data is not available for this category yet.", "plain"))
    with impact_tab:
        panel_row(
            panel(
                "🌍 Environmental impact if improperly disposed",
                field_value(info, "environmentalImpact", "Impact data is not available for this category yet."),
                "danger",
            ),
            panel(
                "🌱 CO₂ savings from recycling",
                field_value(info, "co2Savings", "Data not available."),
                "eco",
            ),
        )
    with disposal_tab:
        show_html(panel(
            "Responsible disposal",
            field_value(info, "disposal", "Keep electronic waste separate from ordinary household waste and follow local collection rules."),
            "plain",
        ))

    show_html(
        '<div class="disclaimer">___________________________________________'
        '_________________________________________'
        '__________________________________________________</div>'
    )


if __name__ == "__main__":
    main()