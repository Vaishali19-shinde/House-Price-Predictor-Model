# Install dependencies (Put this on terminal):
#   pip install streamlit numpy pandas plotly scikit-learn>=1.3.0
#
# To run this code use:
#   streamlit run App.py
#
# If you are facing issue w.r.t cmdlet:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# --------------------------------------------------

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "house.pkl"
FEATURES = ["Area_sqft", "Bedrooms", "Bathrooms", "House_Age", "Floors"]


# -----------------------------
# Enhanced custom styling
# -----------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ═══════════════════════════════════════
           GLOBAL & BACKGROUND
           ═══════════════════════════════════════ */
        * { font-family: 'Inter', sans-serif !important; }

        .stApp {
            background:
                radial-gradient(ellipse at 0% 0%, rgba(6, 182, 212, 0.07), transparent 50%),
                radial-gradient(ellipse at 100% 0%, rgba(16, 185, 129, 0.07), transparent 50%),
                radial-gradient(ellipse at 50% 100%, rgba(99, 102, 241, 0.05), transparent 50%),
                linear-gradient(180deg, #f0fdfa 0%, #f1f5f9 50%, #eef2ff 100%);
            min-height: 100vh;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        /* ═══════════════════════════════════════
           ANIMATIONS
           ═══════════════════════════════════════ */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50%      { transform: translateY(-8px); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse-ring {
            0%   { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.35); }
            70%  { box-shadow: 0 0 0 12px rgba(5, 150, 105, 0); }
            100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
        }

        @keyframes shimmer {
            0%   { background-position: -200% center; }
            100% { background-position: 200% center; }
        }

        @keyframes sparkle {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50%      { opacity: 1; transform: scale(1.5); }
        }

        /* ═══════════════════════════════════════
           HERO CARD
           ═══════════════════════════════════════ */
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #064e3b 30%, #059669 60%, #10b981 100%);
            background-size: 300% 300%;
            animation: gradientShift 8s ease infinite;
            border-radius: 28px;
            padding: 2.5rem 2.8rem;
            color: white;
            box-shadow:
                0 25px 60px rgba(5, 150, 105, 0.15),
                0 10px 20px rgba(15, 23, 42, 0.12),
                inset 0 1px 0 rgba(255,255,255,0.08);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .hero-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }

        .hero-card::after {
            content: '';
            position: absolute;
            bottom: -30%;
            left: 10%;
            width: 250px;
            height: 250px;
            background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
            border-radius: 50%;
            animation: float 8s ease-in-out infinite reverse;
        }

        .hero-card h1 {
            font-family: 'Outfit', sans-serif !important;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
            position: relative;
            z-index: 1;
            letter-spacing: -0.02em;
        }

        .hero-card .hero-subtitle {
            color: #a7f3d0;
            font-size: 1.08rem;
            margin: 0;
            position: relative;
            z-index: 1;
            font-weight: 400;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 50px;
            padding: 6px 16px;
            font-size: 0.82rem;
            color: #d1fae5;
            margin-bottom: 1rem;
            position: relative;
            z-index: 1;
        }

        .hero-badge .dot {
            width: 8px;
            height: 8px;
            background: #34d399;
            border-radius: 50%;
            animation: pulse-ring 2s infinite;
        }

        .hero-stats {
            display: flex;
            gap: 2rem;
            margin-top: 1.2rem;
            position: relative;
            z-index: 1;
        }

        .hero-stat {
            text-align: center;
        }

        .hero-stat-value {
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.6rem;
            font-weight: 700;
            color: white;
        }

        .hero-stat-label {
            font-size: 0.78rem;
            color: #a7f3d0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ═══════════════════════════════════════
           GLASS CARDS
           ═══════════════════════════════════════ */
        .glass-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.45);
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow:
                0 8px 32px rgba(15, 23, 42, 0.06),
                0 2px 8px rgba(15, 23, 42, 0.04);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            animation: fadeInUp 0.6s ease-out;
        }

        /* ═══════════════════════════════════════
           PRESET CARDS
           ═══════════════════════════════════════ */
        .preset-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 0.8rem 0 1.2rem 0;
        }

        .preset-card {
            background: rgba(255,255,255,0.8);
            border: 2px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1.1rem 0.9rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .preset-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, #059669, #10b981);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .preset-card:hover {
            transform: translateY(-4px);
            border-color: #6ee7b7;
            box-shadow: 0 12px 30px rgba(5, 150, 105, 0.12);
        }

        .preset-card:hover::before { opacity: 1; }

        .preset-card.active {
            border-color: #059669;
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            box-shadow: 0 8px 24px rgba(5, 150, 105, 0.15);
        }

        .preset-card.active::before { opacity: 1; }

        .preset-icon { font-size: 2rem; margin-bottom: 0.4rem; }

        .preset-name {
            font-weight: 700;
            font-size: 0.92rem;
            color: #0f172a;
        }

        .preset-desc {
            font-size: 0.76rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* ═══════════════════════════════════════
           RESULT CARD
           ═══════════════════════════════════════ */
        .result-card {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%);
            border: 1px solid #6ee7b7;
            border-radius: 24px;
            padding: 1.6rem 1.5rem;
            text-align: center;
            margin-top: 0.8rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(5, 150, 105, 0.1);
        }

        .result-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                linear-gradient(135deg,
                    transparent 40%,
                    rgba(255,255,255,0.5) 50%,
                    transparent 60%);
            background-size: 200% 200%;
            animation: shimmer 3s ease-in-out infinite;
        }

        .result-tag {
            display: inline-block;
            background: rgba(5, 150, 105, 0.12);
            color: #047857;
            font-weight: 600;
            font-size: 0.78rem;
            padding: 4px 14px;
            border-radius: 50px;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            position: relative;
        }

        .result-value {
            font-family: 'Outfit', sans-serif !important;
            font-size: 3rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1;
            position: relative;
            letter-spacing: -0.02em;
        }

        .result-label {
            color: #475569;
            font-size: 0.92rem;
            margin-top: 0.6rem;
            font-weight: 500;
            position: relative;
        }

        /* ═══════════════════════════════════════
           METRICS
           ═══════════════════════════════════════ */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.85rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            border-color: rgba(5, 150, 105, 0.25);
        }

        div[data-testid="stMetric"] label {
            font-weight: 600 !important;
            color: #475569 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }

        /* ═══════════════════════════════════════
           BUTTONS
           ═══════════════════════════════════════ */
        .stButton > button {
            width: 100%;
            border-radius: 14px;
            min-height: 52px;
            font-weight: 700;
            font-size: 1rem;
            background: linear-gradient(135deg, #059669 0%, #047857 50%, #065f46 100%);
            background-size: 200% 200%;
            color: white;
            border: none;
            letter-spacing: 0.02em;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);
        }

        .stButton > button:hover {
            color: white;
            border: none;
            background-position: 100% 50%;
            box-shadow: 0 8px 30px rgba(5, 150, 105, 0.3);
            transform: translateY(-2px);
        }

        .stButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 2px 8px rgba(5, 150, 105, 0.2);
        }

        /* ═══════════════════════════════════════
           SIDEBAR
           ═══════════════════════════════════════ */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: white !important;
            font-family: 'Outfit', sans-serif !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.15) !important;
        }

        section[data-testid="stSidebar"] .stAlert {
            background: rgba(5, 150, 105, 0.12) !important;
            border: 1px solid rgba(5, 150, 105, 0.25) !important;
            border-radius: 14px !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(239, 68, 68, 0.15) !important;
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
            color: #fca5a5 !important;
            box-shadow: none;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(239, 68, 68, 0.25) !important;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15) !important;
        }

        .sidebar-logo {
            text-align: center;
            font-size: 3.5rem;
            margin-bottom: 0.2rem;
            animation: float 3s ease-in-out infinite;
        }

        .sidebar-brand {
            text-align: center;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #34d399, #6ee7b7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.2rem;
        }

        .sidebar-tagline {
            text-align: center;
            font-size: 0.82rem;
            color: #94a3b8 !important;
            margin-bottom: 1rem;
        }

        .model-status-card {
            background: rgba(5, 150, 105, 0.1);
            border: 1px solid rgba(5, 150, 105, 0.2);
            border-radius: 16px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #34d399;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse-ring 2s infinite;
        }

        .feature-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            margin-bottom: 6px;
            transition: background 0.2s;
        }

        .feature-item:hover { background: rgba(255,255,255,0.08); }

        .feature-icon {
            font-size: 1.1rem;
            width: 28px;
            text-align: center;
        }

        .feature-name { font-weight: 600; font-size: 0.88rem; }
        .feature-desc { font-size: 0.76rem; color: #94a3b8 !important; }

        /* ═══════════════════════════════════════
           SECTION HEADERS
           ═══════════════════════════════════════ */
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 1.5rem 0 1rem 0;
        }

        .section-header .icon-circle {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #059669, #10b981);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.2);
        }

        .section-header h3 {
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.3rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }

        .section-header .section-desc {
            font-size: 0.85rem;
            color: #64748b;
            margin: 0;
        }

        /* ═══════════════════════════════════════
           INSIGHT CARDS
           ═══════════════════════════════════════ */
        .insight-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 0.5rem;
        }

        .insight-item {
            background: rgba(255,255,255,0.7);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 14px;
            padding: 1rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .insight-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            border-color: rgba(5, 150, 105, 0.2);
        }

        .insight-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }

        .insight-title {
            font-weight: 700;
            font-size: 0.88rem;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }

        .insight-text {
            font-size: 0.8rem;
            color: #64748b;
            line-height: 1.4;
        }

        /* ═══════════════════════════════════════
           HISTORY TABLE
           ═══════════════════════════════════════ */
        div[data-testid="stDataFrame"] {
            border-radius: 16px !important;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        }

        /* ═══════════════════════════════════════
           FOOTER
           ═══════════════════════════════════════ */
        .footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            margin-top: 1rem;
        }

        .footer-inner {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 50px;
            padding: 10px 24px;
            color: #64748b;
            font-size: 0.85rem;
        }

        .footer-dot {
            width: 5px;
            height: 5px;
            background: #059669;
            border-radius: 50%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' was not found. Place house.pkl in the same folder as App.py."
        )
    with open(path, "rb") as file:
        obj = pickle.load(file)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def predict_price(model, input_df: pd.DataFrame) -> float:
    try:
        prediction = model.predict(input_df)
    except Exception:
        prediction = model.predict(input_df[FEATURES].to_numpy())
    value = float(np.ravel(prediction)[0])
    if not np.isfinite(value):
        raise ValueError("The model returned an invalid prediction.")
    return value


def price_category(price: float):
    if price < 2500000:
        return "Budget Range", "🏚️", "#ef4444"
    if price < 5000000:
        return "Mid Range", "🏡", "#f59e0b"
    if price < 10000000:
        return "Premium Range", "🏢", "#10b981"
    return "Luxury Range", "🏰", "#6366f1"


def format_inr(value: float) -> str:
    """Format number in Indian numbering system (lakhs/crores)."""
    if value >= 10000000:
        return f"₹ {value / 10000000:.2f} Cr"
    if value >= 100000:
        return f"₹ {value / 100000:.2f} L"
    return f"₹ {value:,.0f}"


def create_gauge(price: float):
    upper_limit = max(20000000, int(np.ceil(price / 5000000) * 5000000))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=price,
            number={"prefix": "₹ ", "font": {"size": 32, "family": "Inter"}},
            title={"text": "Predicted Price", "font": {"size": 14, "color": "#64748b"}},
            gauge={
                "axis": {
                    "range": [0, upper_limit],
                    "tickfont": {"size": 10, "color": "#94a3b8"},
                    "tickprefix": "₹",
                },
                "bar": {"color": "#059669", "thickness": 0.75},
                "bgcolor": "rgba(241,245,249,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 2500000], "color": "rgba(239,68,68,0.08)"},
                    {"range": [2500000, 5000000], "color": "rgba(245,158,11,0.08)"},
                    {"range": [5000000, 10000000], "color": "rgba(16,185,129,0.08)"},
                    {"range": [10000000, upper_limit], "color": "rgba(99,102,241,0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.8,
                    "value": price,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_feature_radar(area, bedrooms, bathrooms, house_age, floors):
    """Create a radar chart showing relative feature strengths."""
    # Normalise each feature to 0-1 range for radar display
    norm_area = min(area / 5000, 1.0)
    norm_bed = min(bedrooms / 6, 1.0)
    norm_bath = min(bathrooms / 5, 1.0)
    norm_age = max(1.0 - house_age / 50, 0.0)  # newer is better
    norm_floors = min(floors / 4, 1.0)

    categories = ["Area", "Bedrooms", "Bathrooms", "Newness", "Floors"]
    values = [norm_area, norm_bed, norm_bath, norm_age, norm_floors]
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(5,150,105,0.12)",
        line=dict(color="#059669", width=2.5),
        marker=dict(size=6, color="#059669"),
        name="Features",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="rgba(148,163,184,0.15)",
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.15)",
                tickfont=dict(size=11, color="#475569", family="Inter"),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=280,
        margin=dict(l=50, r=50, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_price_breakdown(price: float):
    """Create a donut chart showing estimated price breakdown."""
    land_cost = price * 0.40
    construction = price * 0.30
    interiors = price * 0.15
    amenities = price * 0.10
    legal_misc = price * 0.05

    labels = ["Land Cost", "Construction", "Interiors", "Amenities", "Legal & Misc"]
    values = [land_cost, construction, interiors, amenities, legal_misc]
    colors = ["#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=11, family="Inter"),
        hovertemplate="<b>%{label}</b><br>₹ %{value:,.0f}<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font={"family": "Inter"},
        annotations=[
            dict(
                text=f"<b>{format_inr(price)}</b><br><span style='font-size:10px;color:#64748b'>Est. Total</span>",
                x=0.5, y=0.5,
                font=dict(size=16, color="#0f172a", family="Inter"),
                showarrow=False,
            )
        ],
    )
    return fig


def create_history_chart(history_list):
    """Create a sparkline area chart of prediction history."""
    if len(history_list) < 2:
        return None

    prices = [h["Predicted Price"] for h in reversed(history_list)]
    times = [h["Time"] for h in reversed(history_list)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(5,150,105,0.08)",
        line=dict(color="#059669", width=2.5, shape="spline"),
        marker=dict(size=7, color="#059669", line=dict(width=2, color="white")),
        hovertemplate="<b>%{x}</b><br>₹ %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            tickfont=dict(size=10, color="#94a3b8"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.1)",
            showticklabels=True,
            tickfont=dict(size=10, color="#94a3b8"),
            tickprefix="₹ ",
        ),
        font={"family": "Inter"},
    )
    return fig


# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">🏠</div>'
        '<div class="sidebar-brand">House Price AI</div>'
        '<div class="sidebar-tagline">ML-Powered Price Prediction</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Model status
    st.markdown("##### ⚡ Model Status")
    try:
        model = load_model(MODEL_PATH)
        st.markdown(
            '<div class="model-status-card">'
            '<span class="status-dot"></span>'
            '<strong>house.pkl</strong> loaded & ready'
            "</div>",
            unsafe_allow_html=True,
        )
        model_ready = True
    except Exception as error:
        model = None
        model_ready = False
        st.error(str(error))

    st.divider()

    # Feature guide
    st.markdown("##### 📋 Feature Guide")
    features_info = [
        ("📐", "Area_sqft", "Total area in sq ft"),
        ("🛏️", "Bedrooms", "Number of bedrooms"),
        ("🚿", "Bathrooms", "Number of bathrooms"),
        ("📅", "House_Age", "Age in years"),
        ("🏗️", "Floors", "Number of floors"),
    ]
    for icon, name, desc in features_info:
        st.markdown(
            f'<div class="feature-item">'
            f'<span class="feature-icon">{icon}</span>'
            f"<div><div class='feature-name'>{name}</div>"
            f"<div class='feature-desc'>{desc}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick stats
    st.markdown("##### 📊 Session Stats")
    total_predictions = len(st.session_state.history)
    avg_price = (
        np.mean([h["Predicted Price"] for h in st.session_state.history])
        if st.session_state.history
        else 0
    )
    sc1, sc2 = st.columns(2)
    sc1.metric("Predictions", total_predictions)
    sc2.metric("Avg Price", format_inr(avg_price))

    st.divider()

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# -----------------------------
# Hero Header
# -----------------------------
total_preds = len(st.session_state.history)
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-badge">
            <span class="dot"></span>
            ML Model Active
        </div>
        <h1>🏠 House Price Prediction</h1>
        <p class="hero-subtitle">
            Enter property details and get an AI-powered price estimate
            using your trained machine-learning model.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">{total_preds}</div>
                <div class="hero-stat-label">Predictions</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">5</div>
                <div class="hero-stat-label">Features</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{'✓ Ready' if model_ready else '✗ Error'}</div>
                <div class="hero-stat-label">Model</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Presets
# -----------------------------
st.markdown(
    """
    <div class="section-header">
        <div class="icon-circle">⚡</div>
        <div>
            <h3>Quick Presets</h3>
            <p class="section-desc">Select a property profile to auto-fill values</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

presets = {
    "Budget":   {"icon": "🏚️", "desc": "800 sqft · 1 BHK", "Area_sqft": 800,  "Bedrooms": 1, "Bathrooms": 1, "House_Age": 20, "Floors": 1},
    "Standard": {"icon": "🏡", "desc": "1200 sqft · 2 BHK", "Area_sqft": 1200, "Bedrooms": 2, "Bathrooms": 2, "House_Age": 10, "Floors": 1},
    "Premium":  {"icon": "🏢", "desc": "2000 sqft · 3 BHK", "Area_sqft": 2000, "Bedrooms": 3, "Bathrooms": 3, "House_Age": 5,  "Floors": 2},
    "Luxury":   {"icon": "🏰", "desc": "3500 sqft · 5 BHK", "Area_sqft": 3500, "Bedrooms": 5, "Bathrooms": 4, "House_Age": 2,  "Floors": 3},
}

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Standard"

preset_cols = st.columns(4)
for i, (name, info) in enumerate(presets.items()):
    with preset_cols[i]:
        active_class = "active" if st.session_state.selected_preset == name else ""
        st.markdown(
            f"""
            <div class="preset-card {active_class}">
                <div class="preset-icon">{info['icon']}</div>
                <div class="preset-name">{name}</div>
                <div class="preset-desc">{info['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Select {name}", key=f"preset_{name}", use_container_width=True):
            st.session_state.selected_preset = name
            st.rerun()

selected = presets[st.session_state.selected_preset]


# -----------------------------
# Main input and output layout
# -----------------------------
st.markdown(
    """
    <div class="section-header">
        <div class="icon-circle">🏠</div>
        <div>
            <h3>Property Details & Prediction</h3>
            <p class="section-desc">Configure property specifications to get a price estimate</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 📝 Property Specifications")

    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            area_sqft = st.number_input(
                "📐 Area (sq ft)",
                min_value=100,
                max_value=10000,
                value=int(selected["Area_sqft"]),
                step=50,
                help="Total area of the house in square feet.",
            )

        with row1_col2:
            bedrooms = st.number_input(
                "🛏️ Bedrooms",
                min_value=1,
                max_value=10,
                value=int(selected["Bedrooms"]),
                step=1,
                help="Total number of bedrooms.",
            )

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            bathrooms = st.number_input(
                "🚿 Bathrooms",
                min_value=1,
                max_value=10,
                value=int(selected["Bathrooms"]),
                step=1,
                help="Total number of bathrooms.",
            )

        with row2_col2:
            house_age = st.number_input(
                "📅 House Age (years)",
                min_value=0,
                max_value=100,
                value=int(selected["House_Age"]),
                step=1,
                help="Age of the house in years.",
            )

        row3_col1, _ = st.columns(2)

        with row3_col1:
            floors = st.number_input(
                "🏗️ Floors",
                min_value=1,
                max_value=5,
                value=int(selected["Floors"]),
                step=1,
                help="Total number of floors in the house.",
            )

        st.markdown("---")
        st.markdown("##### Input Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Area", f"{area_sqft} sqft")
        m2.metric("Beds", int(bedrooms))
        m3.metric("Baths", int(bathrooms))
        m4.metric("Age", f"{house_age} yrs")
        m5.metric("Floors", int(floors))

        submitted = st.form_submit_button(
            "🔮 Predict House Price",
            disabled=not model_ready,
        )

    # Feature radar chart
    st.markdown("##### 📊 Feature Profile")
    st.plotly_chart(
        create_feature_radar(area_sqft, bedrooms, bathrooms, house_age, floors),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)


input_df = pd.DataFrame(
    [[area_sqft, bedrooms, bathrooms, house_age, floors]],
    columns=FEATURES,
)

prediction = None
prediction_error = None

if submitted:
    try:
        prediction = predict_price(model, input_df)
        category, icon, _ = price_category(prediction)

        st.session_state.history.insert(
            0,
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Area_sqft": int(area_sqft),
                "Bedrooms": int(bedrooms),
                "Bathrooms": int(bathrooms),
                "House_Age": int(house_age),
                "Floors": int(floors),
                "Predicted Price": round(prediction, 2),
                "Category": category,
            },
        )

        st.session_state.history = st.session_state.history[:20]

    except Exception as error:
        prediction_error = str(error)


with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 💰 Prediction Result")

    if prediction_error:
        st.error(f"Prediction failed: {prediction_error}")

    elif prediction is not None:
        category, icon, cat_color = price_category(prediction)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-tag">{icon} {category}</div>
                <div class="result-value">₹ {prediction:,.2f}</div>
                <div class="result-label">Predicted House Price</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            create_gauge(prediction),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Key metrics
        monthly_emi = prediction / 240 if prediction > 0 else 0
        price_per_sqft = prediction / area_sqft if area_sqft > 0 else 0

        r1, r2, r3 = st.columns(3)
        r1.metric("Total Price", format_inr(prediction))
        r2.metric("₹/sq ft", f"₹ {price_per_sqft:,.0f}")
        r3.metric("EMI (20yr)", format_inr(monthly_emi))

        # Price breakdown donut
        st.markdown("##### 📋 Estimated Cost Breakdown")
        st.plotly_chart(
            create_price_breakdown(prediction),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    else:
        st.markdown(
            """
            <div style="text-align:center; padding:2rem 1rem;">
                <div style="font-size:4rem; margin-bottom:0.5rem; animation: float 3s ease-in-out infinite;">🏠</div>
                <div style="font-size:1.1rem; font-weight:600; color:#0f172a; margin-bottom:0.3rem;">
                    Ready to Predict
                </div>
                <div style="font-size:0.88rem; color:#64748b;">
                    Enter property details on the left and click<br>
                    <strong>Predict House Price</strong> to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            create_gauge(0),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Insights section
# -----------------------------
st.markdown(
    """
    <div class="section-header">
        <div class="icon-circle">💡</div>
        <div>
            <h3>Prediction Insights</h3>
            <p class="section-desc">Key factors that influence house price predictions</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="insight-grid">
        <div class="insight-item">
            <div class="insight-icon">📐</div>
            <div class="insight-title">Area (sq ft)</div>
            <div class="insight-text">Typically the strongest predictor. Larger homes command significantly higher prices.</div>
        </div>
        <div class="insight-item">
            <div class="insight-icon">🛏️</div>
            <div class="insight-title">Bedrooms & Bathrooms</div>
            <div class="insight-text">More rooms add value proportionally. A balanced bed-to-bath ratio is key.</div>
        </div>
        <div class="insight-item">
            <div class="insight-icon">📅</div>
            <div class="insight-title">House Age</div>
            <div class="insight-text">Newer constructions generally command premium prices due to modern amenities.</div>
        </div>
        <div class="insight-item">
            <div class="insight-icon">🏗️</div>
            <div class="insight-title">Floors</div>
            <div class="insight-text">Multi-storey homes tend to be valued higher, offering more usable space.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Model input preview
st.markdown(
    """
    <div class="section-header">
        <div class="icon-circle">📄</div>
        <div>
            <h3>Model Input Preview</h3>
            <p class="section-desc">Raw feature values sent to the ML model</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(
    input_df,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# Prediction history
# -----------------------------
st.markdown(
    """
    <div class="section-header">
        <div class="icon-circle">📜</div>
        <div>
            <h3>Prediction History</h3>
            <p class="section-desc">Track and compare your recent predictions</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.history:
    # Sparkline chart
    history_chart = create_history_chart(st.session_state.history)
    if history_chart:
        st.plotly_chart(
            history_chart,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="house_price_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.metric("Total Predictions", len(st.session_state.history))
else:
    st.markdown(
        """
        <div style="text-align:center; padding:2rem; color:#94a3b8;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📊</div>
            <div style="font-weight:600; color:#64748b;">No predictions yet</div>
            <div style="font-size:0.85rem;">Make your first prediction to start tracking history</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        <div class="footer-inner">
            🏠 House Price Prediction Dashboard
            <span class="footer-dot"></span>
            Built with Streamlit & Plotly
            <span class="footer-dot"></span>
            ML Powered
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
