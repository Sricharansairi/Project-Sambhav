import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.styles import load_css, nav_html, disclaimer_html

st.set_page_config(
    page_title="Sambhav — Multi-Modal Probabilistic Inference Engine",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(load_css(), unsafe_allow_html=True)
st.markdown(nav_html(""), unsafe_allow_html=True)

st.markdown("""
<div class="page-content">

<!-- ── HERO SECTION ─────────────────────────────── -->
<div style="
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    position: relative;
    padding: 0 40px;
">

<!-- Geometric circles -->
<svg style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;z-index:0;opacity:0.06;" width="740" height="740" viewBox="0 0 740 740">
    <circle cx="370" cy="370" r="360" fill="none" stroke="white" stroke-width="1"/>
    <circle cx="370" cy="370" r="290" fill="none" stroke="white" stroke-width="1"/>
    <line x1="180" y1="0" x2="180" y2="740" stroke="white" stroke-width="1"/>
    <line x1="560" y1="0" x2="560" y2="740" stroke="white" stroke-width="1"/>
</svg>

<!-- Left side stats -->
<div style="position:absolute;left:40px;top:50%;transform:translateY(-60%);">
    <div style="margin-bottom:48px;">
        <div class="side-stat-value">12</div>
        <div class="side-stat-label">Operating Modes</div>
    </div>
    <div>
        <div class="side-stat-value">18</div>
        <div class="side-stat-label">Core Features</div>
    </div>
</div>

<!-- Right side stats -->
<div style="position:absolute;right:40px;top:50%;transform:translateY(-60%);text-align:right;">
    <div style="margin-bottom:48px;">
        <div class="side-stat-value" style="font-family:'JetBrains Mono',monospace;">&lt;0.12</div>
        <div class="side-stat-label">Target Brier Score</div>
    </div>
    <div>
        <div class="side-stat-value">5</div>
        <div class="side-stat-label">Input Modalities</div>
    </div>
</div>

<!-- Center content -->
<div style="position:relative;z-index:1;max-width:800px;">

    <!-- Eyebrow pill -->
    <div class="eyebrow-pill" style="margin-bottom:32px;display:inline-flex;">
        <div class="eyebrow-pill-dot"></div>
        <span class="eyebrow-pill-text">MULTI-MODAL PROBABILISTIC INFERENCE ENGINE</span>
    </div>

    <!-- Hero headline -->
    <div class="hero-h1" style="font-size:clamp(48px,7vw,96px);margin-bottom:0;">Don't predict.</div>
    <div class="hero-h2" style="font-size:clamp(40px,6vw,72px);margin-bottom:28px;">
        Measure <span class="hero-accent">Possibility.</span>
    </div>

    <!-- Description -->
    <p class="body-large" style="max-width:560px;margin:0 auto 36px;">
        Sambhav outputs calibrated probability distributions — not labels.<br>
        Every number explained. Every prediction audited. Zero black box.
    </p>

    <!-- CTA buttons -->
    <div style="display:flex;gap:16px;justify-content:center;margin-bottom:40px;flex-wrap:wrap;">
        <button class="btn-primary" onclick="window.location.href='/Dashboard'">Start a Prediction</button>
        <button class="btn-secondary">View Documentation</button>
    </div>

    <!-- Mode chips -->
    <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
        <span class="chip active">Guided</span>
        <span class="chip">Free Inference</span>
        <span class="chip">Document</span>
        <span class="chip">Comparative</span>
        <span class="chip">Fact-Check</span>
    </div>

</div>
</div>

<!-- ── STATS STRIP ───────────────────────────────── -->
<div style="padding:0 40px;">
<div class="stats-strip">
    <div class="stat-cell">
        <div class="stat-value">12</div>
        <div class="stat-label">Operating Modes</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value">18</div>
        <div class="stat-label">Core Features</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value" style="font-family:'JetBrains Mono',monospace;">&lt;0.12</div>
        <div class="stat-label">Target Brier Score</div>
    </div>
    <div class="stat-cell">
        <div class="stat-value">5</div>
        <div class="stat-label">Input Modalities</div>
    </div>
</div>
</div>

<!-- ── FEATURE ROWS ──────────────────────────────── -->
<div style="padding:0 40px;margin-top:20px;">

    <div class="feature-row">
        <div style="display:flex;align-items:flex-start;gap:20px;width:50%;">
            <span class="feature-number">01</span>
            <div>
                <div class="feature-title">Seven-Stage Prediction Pipeline</div>
                <div class="feature-desc">Input parse → domain detect → param audit → ML ensemble → LLM cross-val → confidence audit → output.</div>
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:20px;width:50%;">
            <span class="feature-number">02</span>
            <div>
                <div class="feature-title">Real-Time Reliability Index</div>
                <div class="feature-desc">Live 0-100% trust score before results. SHAP-powered — shows exactly which inputs to add.</div>
            </div>
        </div>
    </div>

    <div class="feature-row">
        <div style="display:flex;align-items:flex-start;gap:20px;width:50%;">
            <span class="feature-number">03</span>
            <div>
                <div class="feature-title">Three-Engine Audit System</div>
                <div class="feature-desc">Engine 1 audits inputs. Engine 2 audits outputs. Engine 3 audits certainty. Seven flag types.</div>
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:20px;width:50%;">
            <span class="feature-number">04</span>
            <div>
                <div class="feature-title">Dual-LLM Fact-Check Module</div>
                <div class="feature-desc">Groq/Llama for deep historical knowledge. xAI Grok for real-time discourse. Weighted average.</div>
            </div>
        </div>
    </div>

</div>

<!-- bottom spacer for disclaimer -->
<div style="height:40px;"></div>

</div>
""", unsafe_allow_html=True)

st.markdown(disclaimer_html(), unsafe_allow_html=True)
