# 🌌 Project Sambhav: The Unified Multi-Modal Probabilistic Inference Ecosystem

> **Uncertainty, Quantified. Complexity, Calibrated.**

Project Sambhav is a frontier-grade research framework and production-ready application designed to transform raw, high-entropy data into high-fidelity, calibrated probabilities. In an era where "black box" AI provides answers without justification, Sambhav provides **inference with evidence**, **predictions with reliability**, and **decisions with accountability**.

---

## 📖 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Vision & Philosophy](#-vision--philosophy)
3. [Core Architectural Framework](#-core-architectural-framework)
    - [The Dual-Layer Inference Model](#the-dual-layer-inference-model)
    - [Multi-Agent Adversarial Framework](#multi-agent-adversarial-framework)
    - [Universal Document Pipeline](#universal-document-pipeline)
4. [Technical Stack Deep-Dive](#-technical-stack-deep-dive)
    - [Backend (FastAPI Engine)](#backend-fastapi-engine)
    - [Frontend (React High-Performance UI)](#frontend-react-high-performance-ui)
    - [LLM Orchestration (Groq & NVIDIA NIM)](#llm-orchestration-groq--nvidia-nim)
5. [Operational Modes](#-operational-modes)
    - [Conversational Mode](#conversational-mode)
    - [Adversarial Stress-Testing](#adversarial-stress-testing)
    - [Comparative Scenarios](#comparative-scenarios)
6. [Security & Ethics](#-security--ethics)
    - [Authentication & Data Integrity](#authentication--data-integrity)
    - [The Fail-Safe Audit System](#the-fail-safe-audit-system)
7. [Installation & Deployment](#-installation--deployment)
8. [Module-by-Module Reference](#-module-by-module-reference)
9. [Future Roadmap (The P.2030 Vision)](#-future-roadmap-the-p2030-vision)
10. [Legal & Attribution](#-legal--attribution)

---

## 🚀 Executive Summary

Project Sambhav is not just a tool; it is an **Inference Ecosystem**. It addresses the "Confidence Crisis" in modern AI by moving away from binary "Yes/No" answers and toward a nuanced, probabilistic understanding of reality. 

By combining **Machine Learning (ML)** for pattern recognition and **Large Language Models (LLM)** for semantic reasoning, Sambhav achieves a level of predictive calibration previously reserved for high-frequency trading floors and military intelligence units.

---

## 🎯 Vision & Philosophy

The name **Sambhav** (derived from Sanskrit for "Possible") reflects our core belief: *Everything is a distribution of possibilities.*

Our philosophy is built on four pillars:
1.  **Calibration over Confidence**: It is better to be 70% sure and right 70% of the time than to be 100% sure and wrong.
2.  **Multi-Modal Truth**: Truth is rarely found in a single source. Sambhav integrates text, code, images, and large-scale documentation to find the "common thread."
3.  **Adversarial Rigor**: We don't just ask the AI for an answer; we force multiple AI agents to argue against each other to find the truth.
4.  **Radical Transparency**: Every prediction comes with a **SHAP Explanation**, a **Reliability Index**, and an **Audit Log**.

---

## 🏗️ Core Architectural Framework

### The Dual-Layer Inference Model
Sambhav operates on a **Synchronous Multi-Path Pipeline**:
- **Layer 1: Pattern Recognition (ML)**: Uses XGBoost and LightGBM models trained on curated datasets (HR, Finance, Health, Student Performance). These models provide the "cold, hard stats."
- **Layer 2: Semantic Reasoning (LLM)**: High-speed inference via Groq (Llama 3.1/3.3) and NVIDIA NIM (Kimi K2.5). This layer interprets the context, identifies nuances, and detects "soft" signals that pure numbers miss.
- **The Reconciliation**: The `core/predictor.py` engine compares both layers. If they disagree, the **Reliability Index** drops, and the **Audit System** flags the prediction for review.

### Multi-Agent Adversarial Framework
Located in `llm/multi_agent.py`, this system spawns four distinct "personalities" for every complex prediction:
- **The Optimist**: Looks for the best-case scenario and positive growth signals.
- **The Pessimist**: Identifies systemic risks, failure points, and tail-end risks.
- **The Realist**: Balances both views based on historical data.
- **The Devil's Advocate**: Specifically tries to find "physiological impossibilities" or logical fallacies in the input data.

### Universal Document Pipeline
In `vision/document_pipeline.py`, we utilize the **NVIDIA NIM Kimi K2.5** engine. 
- **1 Million Token Window**: Unlike standard RAG systems that chunk data, Sambhav reads the **entire document** (up to 1,000,000 tokens) in one pass.
- **Universal Support**: Processes PDF, DOCX, TXT, CSV, and even source code (`.py`, `.ts`, `.js`) to extract predictive parameters.

---

## 🛠️ Technical Stack Deep-Dive

### Backend (FastAPI Engine)
The backbone of Sambhav is a high-concurrency FastAPI server.
- **Endpoints**: Modular routing for Auth, Prediction, Vision, and Fact-Checking.
- **Database**: SQLAlchemy with PostgreSQL support (locally falling back to SQLite for edge deployment).
- **Rate Limiting**: Custom token-bucket rate limiting (`api/rate_limiter.py`) to manage API costs.

### Frontend (React High-Performance UI)
A futuristic, dark-mode interface built for speed and clarity.
- **Component Library**: Custom Glassmorphism UI components built with Tailwind CSS and Framer Motion.
- **Visualizations**: Real-time SHAP charts and probability distributions.
- **State Management**: Lightweight React hooks with optimized re-renders for real-time model response generation.

### LLM Orchestration
- **Groq Llama 3.1/3.3**: Used for sub-second reasoning and probability generation.
- **NVIDIA NIM**: Used for vision-language tasks and large-document ingestion.
- **Key Rotator**: Located in `api/key_rotator.py`, this system automatically switches between API keys if limits are reached, ensuring zero downtime.

---

## 🎮 Operational Modes

### Conversational Mode
For users who don't have structured data. The system engages in a guided dialogue, dynamically generating questions based on the selected domain to "fill in the blanks" of the predictive model.

### Adversarial Stress-Testing
Users can toggle "Adversarial Mode" to see how the system handles contradictory or malicious inputs. This is crucial for verifying the **Safety Layer** (`core/safety.py`).

### Comparative Scenarios
Allows users to run "What-If" simulations. *What happens to the probability if I change Parameter X by 20%?* The system re-runs the entire dual-layer inference and provides a delta analysis.

---

## 🔒 Security & Ethics

### Authentication & Data Integrity
- **Bcrypt Hashing**: Passwords never touch the database in plain text.
- **JWT tokens**: Secure, stateless session management.
- **Data Export**: Users can download their entire prediction history in a structured JSON format for auditability.

### The Fail-Safe Audit System
Every prediction is screened for:
1.  **Physiological Impossibilities**: (e.g., a student with a -5% GPA).
2.  **Adversarial Attacks**: Attempts to "jailbreak" the LLM logic.
3.  **Low Confidence**: If the ML and LLM layers have a variance greater than 30%, the system forces a "Low Reliability" warning.

---

## 🚦 Installation & Deployment

### 1. Backend: HuggingFace Spaces (Docker SDK)
- **Host**: HuggingFace Spaces (Private/Public repo with Docker SDK).
- **Setup**: 
    1.  Create a new Space with the **Docker SDK**.
    2.  Set these **Secrets** in the Space settings:
        - `GROQ_API_KEY`: For fast inference.
        - `NVIDIA_API_KEY`: For large-context document analysis.
        - `DATABASE_URL`: Your Supabase PostgreSQL connection string.
        - `ALLOW_ORIGINS`: Your Vercel domain (e.g., `https://sambhav.vercel.app`).
    3.  Push the project code. HF will build using the root `Dockerfile`.

### 2. Frontend: Vercel (React + Vite)
- **Host**: Vercel.
- **Setup**:
    1.  Connect your GitHub repository to Vercel.
    2.  Set **Root Directory** to `frontend/`.
    3.  Set **Framework Preset** to `Vite`.
    4.  Set **Environment Variables**:
        - `VITE_API_URL`: The URL of your HuggingFace Space (e.g., `https://sricharansairi-project-sambhav.hf.space/api`).
    5.  Deploy. Vercel will handle the rest with zero-config using the `frontend/vercel.json` for SPA routing.

### 3. Database: Supabase PostgreSQL
- **Host**: Supabase (Free Tier).
- **Setup**: Use the `alembic` migrations in the `db/` folder to set up your production schema.

---

## 📂 Module-by-Module Reference

- `api/`: REST interface and security middleware.
- `core/`: The "Brain" – contains the predictor, reliability logic, and SHAP engine.
- `db/`: Data persistence and user modeling.
- `llm/`: Multi-model orchestration and adversarial agent logic.
- `vision/`: File ingestion and multi-modal signal extraction.
- `reports/`: Automated PDF/XLSX/PPTX report generation.
- `schemas/`: The source of truth for all domain parameters.

---

## 🔮 Future Roadmap (The P.2030 Vision)

- **P.01 (Current)**: Stable multi-modal inference with 1M token support.
- **P.02 (Q3 2026)**: Integration of real-time web-search signals into the prediction tree.
- **P.03 (2027)**: On-device "Small Language Model" (SLM) fallback for offline predictions.
- **P.10 (2030)**: Fully autonomous "Continuous Calibration" – the system self-corrects based on real-world outcomes reported by users.

---

## ⚖️ Legal & Attribution

Project Sambhav is provided for research and informational purposes. While we strive for extreme calibration, the system is probabilistic by nature. 

**Developed with ❤️ by Sricharan Sairi.**
**Powered by the NVIDIA NIM & Groq Ecosystem.**

---
*End of Document. Project Sambhav — Uncertainty, Quantified.*
