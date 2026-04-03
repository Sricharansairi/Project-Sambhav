# Project Sambhav: Multi-Modal Probabilistic Inference Engine

Project Sambhav is a high-reliability, research-grade AI framework designed to transform complex predictions into calibrated probabilities. It leverages a dual-layer architecture (ML + LLM) to quantify uncertainty across multiple domains and modalities including text, images, video, and large-context documents.

## 🚀 Key Features

### 1. Multi-Modal Analysis
- **Universal Document Analysis**: Process PDFs, DOCX, TXT, and code files using NVIDIA NIM Kimi K2.5 with a **1 million token context window**.
- **Visual & Audio Cues**: Extract predictive signals from images, videos, and voice recordings.
- **Conversational Mode**: An expert-guided interface that asks relevant questions to build a high-reliability prediction.

### 2. Calibrated Uncertainty Quantification
- **Dual-Layer Prediction**: Combines traditional Machine Learning (XGBoost, LightGBM) with high-speed LLM reasoning (Groq/Llama 3.1).
- **Reliability Index**: A 0.0–1.0 score indicating the trustworthiness of a prediction based on parameter completeness and model agreement.
- **Isotonic Calibration**: ML models are calibrated using Isotonic Regression to prevent overconfidence.

### 3. Advanced Operating Modes
- **Adversarial Mode**: Stress-test the system with extreme or contradictory inputs to trigger the fail-safe audit system.
- **What-If Stories**: Generate branching probability trees to explore different future scenarios.
- **Comparative Inference**: Compare probabilities across multiple sets of input parameters simultaneously.
- **Expert Consultation**: A multi-agent debate (Optimist, Pessimist, Realist, Devil's Advocate) to reconcile conflicting signals.

### 4. Data Transparency & Management
- **SHAP Explanations**: Visual breakdown of which parameters contributed most to the final probability.
- **Audit System**: Real-time flagging of adversarial inputs, physiological impossibilities, and significant ML-LLM disagreements.
- **Data Privacy**: Local history management with options to export all data in JSON format or clear history for GDPR/DPDP compliance.

## 🛠️ Technology Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Lucide Icons, Framer Motion (Motion).
- **Backend**: FastAPI (Python 3.10+), SQLAlchemy ORM, PostgreSQL (Supabase).
- **Inference Engines**:
    - **ML**: Scikit-Learn, XGBoost, LightGBM.
    - **LLM**: Groq (Llama 3.1/3.3), NVIDIA NIM (Kimi, Qwen), Google Gemini.
- **Reporting**: PDF, DOCX, XLSX, and PPTX report generation.

## 📦 Project Structure

```text
├── api/                # FastAPI endpoints and middleware
├── core/               # Main prediction pipeline and model logic
├── db/                 # Database models and operations
├── frontend/           # React + Vite frontend application
├── llm/                # Multi-model LLM clients and routing logic
├── models/             # ML model metadata and training scripts
├── schemas/            # Domain definitions and parameter registries
├── vision/             # Multi-modal processing pipelines (Doc, Image, Video, Voice)
└── reports/            # Multi-format report generation modules
```

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- API Keys for: Groq, NVIDIA NIM, and (optional) Google Gemini.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sricharansairi/Project-Sambhav.git
   cd Project-Sambhav
   ```

2. **Setup Backend**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # Add your API keys here
   python api/main.py
   ```

3. **Setup Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 🔒 Security & Compliance
- **Strong Hashing**: All user passwords are secured using bcrypt.
- **Token-Based Auth**: Stateless authentication using JWT.
- **Robust .gitignore**: Ensures sensitive API keys, local databases, and large model binaries are never committed to version control.

---
**Disclaimer**: Sambhav provides probabilistic estimates, not certainties. Always verify critical predictions with domain experts.
