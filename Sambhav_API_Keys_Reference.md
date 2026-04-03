# 🔑 PROJECT SAMBHAV — API KEYS REFERENCE
## Complete Guide: What Keys, How Many, Where to Get Them
### Session Date: March 17, 2026 | For Tomorrow's Session

---

## 📊 CURRENT KEY STATUS (from .env)

| Service | Keys in .env | Status | Issue |
|---|---|---|---|
| Groq | 3 keys | ✅ WORKING (2/3) | Key 1 was invalid, replaced |
| NVIDIA NIM | 2 keys | ✅ WORKING | Good |
| Google Gemini | 6 keys | ⚠️ SAME ACCOUNT | Shared quota |
| Google Custom Search | 10 keys | 🔴 BROKEN | Wrong key type (401 error) |
| NewsAPI | 2 keys | ✅ WORKING | Good |
| GNews | 1 key | ✅ WORKING | Good |
| Cohere | 2 keys | ✅ WORKING | Good |
| Brave Search | 0 keys | ❌ NOW PAID | Skip — use DuckDuckGo |
| OpenWeatherMap | 0 keys | ⏳ OPTIONAL | Get later |
| Whisper | 0 keys | ⏳ PHASE 8 | Get later |

---

## 🎯 WHAT THE DOCUMENTATION SAYS IS NEEDED

### 1. 🤖 GROQ — Primary LLM (Llama 3.3 70B)
- **Role:** Main LLM inference, multi-agent debate, free inference mode
- **Free Limit:** 14,400 requests/day per key
- **Keys Needed:** 5–10 (from DIFFERENT accounts for separate quotas)
- **Current:** 3 keys (2 working) ← **Need 2-7 more**
- **Get from:** https://console.groq.com
- **Steps:**
  1. Sign up with different email accounts
  2. Go to API Keys → Create API Key
  3. Add as `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc.
- **Env variable names:** `GROQ_API_KEY_1` through `GROQ_API_KEY_10`

---

### 2. 🖥️ NVIDIA NIM — Vision + Document LLM
- **Role:** Qwen VLM (image), Nemotron VL (video), Kimi K2.5 (documents)
- **Free Limit:** Free tier credits (limited)
- **Keys Needed:** 3–5 (different accounts)
- **Current:** 2 keys ← **Need 1-3 more**
- **Get from:** https://build.nvidia.com
- **Steps:**
  1. Sign up at build.nvidia.com
  2. Go to API Keys → Generate Key
  3. Add as `NVIDIA_API_KEY_1`, `NVIDIA_API_KEY_2`, etc.
- **Env variable names:** `NVIDIA_API_KEY_1` through `NVIDIA_API_KEY_5`

---

### 3. 🔮 GOOGLE GEMINI — Vision Fallback
- **Role:** Vision fallback when NVIDIA fails, document analysis
- **Free Limit:** 1,500 req/day per account
- **Keys Needed:** 5–10 (MUST be from DIFFERENT Google accounts)
- **Current:** 6 keys but ALL SAME ACCOUNT ← **Critical issue!**
- **Get from:** https://aistudio.google.com
- **Steps:**
  1. Sign in with DIFFERENT Google accounts (use different emails)
  2. Go to Get API Key → Create API Key
  3. Each account gives separate 1,500/day quota
- **Env variable names:** `GEMINI_API_KEY_1` through `GEMINI_API_KEY_10`
- **⚠️ WARNING:** Same account = same quota! Need different Google accounts

---

### 4. 🔍 GOOGLE CUSTOM SEARCH — Fact-Check Web Search
- **Role:** Primary web search for fact-checking
- **Free Limit:** 100 queries/key/day
- **Keys Needed:** 10+ (from DIFFERENT Google accounts)
- **Current:** 10 keys but ALL 401 BROKEN ← **Must fix!**
- **Root Cause:** Wrong key type — need Custom Search JSON API key, not regular API key

#### ✅ HOW TO FIX (IMPORTANT):

**Step 1 — Create Search Engine ID:**
1. Go to https://programmablesearchengine.google.com
2. Click "Add" → Name it "Sambhav Search"
3. Set "Search the entire web" = ON
4. Copy the **Search Engine ID (cx)**
5. Add to `.env` as: `GOOGLE_SEARCH_ENGINE_ID=your_cx_here`

**Step 2 — Get proper API Key:**
1. Go to https://console.cloud.google.com
2. Create a project (or use existing)
3. Go to APIs & Services → Library
4. Search "Custom Search JSON API" → Enable it
5. Go to APIs & Services → Credentials → Create Credentials → API Key
6. Copy the key → Add as `GOOGLE_SEARCH_API_KEY_1`
7. Repeat with different Google accounts for more keys

- **Env variable names:** `GOOGLE_SEARCH_API_KEY_1` through `GOOGLE_SEARCH_API_KEY_10`
- **Also needed:** `GOOGLE_SEARCH_ENGINE_ID=your_cx_id`

---

### 5. 📰 NEWSAPI — News for Fact-Check
- **Role:** News search fallback for fact-checking
- **Free Limit:** 100 req/day per key
- **Keys Needed:** 10+ (different accounts/emails)
- **Current:** 2 keys ← **Need 8 more**
- **Get from:** https://newsapi.org
- **Steps:**
  1. Sign up with different emails
  2. Go to Account → API Key → Copy
- **Env variable names:** `NEWS_API_KEY_1` through `NEWS_API_KEY_10`

---

### 6. 📡 GNEWS API — News Fallback
- **Role:** Secondary news search fallback
- **Free Limit:** 100 req/day per key
- **Keys Needed:** 5
- **Current:** 1 key ← **Need 4 more**
- **Get from:** https://gnews.io
- **Steps:** Sign up → Dashboard → API Key
- **Env variable names:** `GNEWS_API_KEY_1` through `GNEWS_API_KEY_5`

---

### 7. 🌤️ OPENWEATHERMAP — Context Injection (Optional)
- **Role:** Weather context for predictions (optional feature)
- **Free Limit:** 60 calls/min (very generous)
- **Keys Needed:** 2
- **Current:** 0 keys ← **Get when building context injection**
- **Get from:** https://openweathermap.org/api
- **Env variable names:** `OPENWEATHERMAP_API_KEY_1`, `OPENWEATHERMAP_API_KEY_2`

---

### 8. 🎤 WHISPER — Voice to Text (Phase 8)
- **Role:** Voice input transcription
- **Free Limit:** OpenAI free tier
- **Keys Needed:** 3
- **Current:** 0 keys ← **Get when testing voice pipeline**
- **Get from:** https://platform.openai.com
- **Env variable names:** `WHISPER_API_KEY_1` through `WHISPER_API_KEY_3`
- **Note:** Same as OpenAI API key

---

### 9. ~~BRAVE SEARCH~~ → REPLACED BY DUCKDUCKGO
- **Status:** Now paid — skip entirely
- **Replacement:** DuckDuckGo (free, no key needed, already implemented)
- **Library:** `duckduckgo-search` (already installed)

---

## 📋 PRIORITY ORDER FOR TOMORROW

### 🔴 CRITICAL — Fix first:
1. **Google Custom Search** — Fix the 401 error (wrong key type)
   - Create Search Engine ID at programmablesearchengine.google.com
   - Get proper Custom Search JSON API keys from console.cloud.google.com
2. **Groq** — Get 2-7 more keys from different accounts (main LLM!)

### 🟡 IMPORTANT — Get soon:
3. **Google Gemini** — Get keys from different Google accounts (not same!)
4. **NewsAPI** — Get 8 more keys for fact-check fallback

### 🟢 OPTIONAL — Get when needed:
5. **NVIDIA NIM** — Get 1-3 more keys
6. **GNews** — Get 4 more keys
7. **OpenWeatherMap** — Get when building context injection
8. **Whisper** — Get when testing voice pipeline

---

## 📝 COMPLETE .env TEMPLATE (Target State)

```
# ── GROQ (5-10 keys, different accounts) ──────────────────────
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...
GROQ_API_KEY_5=gsk_...

# ── NVIDIA NIM (3-5 keys) ──────────────────────────────────────
NVIDIA_API_KEY_1=nvapi-...
NVIDIA_API_KEY_2=nvapi-...
NVIDIA_API_KEY_3=nvapi-...

# ── GOOGLE GEMINI (5-10 keys, DIFFERENT Google accounts!) ──────
GEMINI_API_KEY_1=AIza...
GEMINI_API_KEY_2=AIza...  ← different account
GEMINI_API_KEY_3=AIza...  ← different account
GEMINI_API_KEY_4=AIza...  ← different account
GEMINI_API_KEY_5=AIza...  ← different account

# ── GOOGLE CUSTOM SEARCH (fix 401 first!) ─────────────────────
GOOGLE_SEARCH_ENGINE_ID=your_cx_id_here  ← from programmablesearchengine.google.com
GOOGLE_SEARCH_API_KEY_1=AIza...  ← Custom Search JSON API key (NOT regular API key)
GOOGLE_SEARCH_API_KEY_2=AIza...
GOOGLE_SEARCH_API_KEY_3=AIza...
...up to 10

# ── NEWSAPI (10 keys, different emails) ───────────────────────
NEWS_API_KEY_1=...
NEWS_API_KEY_2=...
...up to 10

# ── GNEWS (5 keys) ────────────────────────────────────────────
GNEWS_API_KEY_1=...
...up to 5

# ── OPENWEATHERMAP (optional, 2 keys) ─────────────────────────
OPENWEATHERMAP_API_KEY_1=...
OPENWEATHERMAP_API_KEY_2=...

# ── WHISPER / OPENAI (3 keys, phase 8) ───────────────────────
WHISPER_API_KEY_1=sk-...
WHISPER_API_KEY_2=sk-...
WHISPER_API_KEY_3=sk-...

# ── APP SETTINGS ──────────────────────────────────────────────
SECRET_KEY=sambhav_secret_change_this_in_production
DATABASE_URL=sqlite:///sambhav.db
FASTAPI_URL=http://localhost:8000
```

---

## 🚀 WHERE WE LEFT OFF — TOMORROW'S STARTING POINT

### Approach 1 (Phase Building) Status:
- ✅ Phase 7 — LLM Integration (groq_client, multi_agent, nvidia_client, predictor)
- ✅ Phase 8 — Vision Pipeline (image, video, document, voice)
- ✅ Phase 9 — Audit System (audit_system, reliability_index, safety)
- ✅ Phase 10 — Fact-Check Module (fact_checker)
- 🔄 **Phase 11 — FastAPI Backend** ← RESUME HERE
  - ✅ `api/main.py`
  - ✅ `api/endpoints/predict.py`
  - ✅ `api/endpoints/factcheck.py`
  - ✅ `api/endpoints/vision.py`
  - ✅ `api/endpoints/auth.py`
  - ✅ `api/endpoints/history.py`
  - ✅ `api/__init__.py`
  - ✅ `api/endpoints/__init__.py`
  - 🔄 `api/key_rotator.py` ← was building when we stopped
  - ❌ `api/rate_limiter.py` ← not built yet
  - ❌ Test FastAPI server ← not done yet

### Approach 2 (Model Training) Status:
- ✅ Student      → student_stacking_v3.joblib       Brier: 0.0932
- ✅ Higher Edu   → student_dropout_stacking_v4.joblib Brier: 0.0833
- ✅ HR Attrition → hr_stacking_v3.joblib             Brier: 0.0400 🔥🔥
- ✅ Disease      → disease_stacking_v2.joblib         Brier: 0.0818
- ✅ Fitness      → fitness_stacking_v2.joblib         Brier: 0.0659 🔥
- ✅ Behavioral   → behavioral_stacking_v5.joblib      Brier: 0.0571 🔥
- ⚠️ Loan         → loan_stacking_v4.joblib            Brier: 0.1263 (data ceiling)
- ⚠️ Mental Health→ mental_health_stacking_v5.joblib   Brier: 0.1370 (data ceiling)
- ⚠️ Claim        → claim_stacking_v5.joblib           Brier: 0.1672 (LIAR dataset noise)

### LLM Calibration Status:
- ✅ Direction accuracy: 10/10 = 100%
- ✅ Calibration score: 9/10 = 90%
- ✅ Few-shot examples added for Student, HR, Disease, Loan

---

## ⚠️ IMPORTANT NOTES FOR TOMORROW

1. **key_rotator.py** — was half-written, needs to be completed before testing FastAPI
2. **Google Search keys** — MUST fix before fact-check works properly
3. **domain_registry.yaml** — already updated with latest model paths
4. **NEVER use n_jobs=-1** on Mac — always n_jobs=2
5. **NEVER train on deployment server** — only inference
6. **Behavioral model** — verified Brier 0.057 on real data (leakage fix confirmed)
7. **LLM base rates injected** into prompts — don't remove this
8. **Few-shot examples** in groq_client.py — critical for calibration

---

*Sambhav may be incorrect. Always verify important decisions independently.*
*Document created: March 17, 2026 | Sri Indu Institute of Engineering & Technology*
