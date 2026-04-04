import os, sys, time, logging, json
from datetime import datetime
from dotenv import load_dotenv

# Get absolute path of project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()
logger = logging.getLogger(__name__)

# Use /tmp for logs in production, project root in dev
BASE = PROJECT_ROOT
LOG_FILE = os.path.join(os.getenv("EXPORTS_DIR", "/tmp"), "key_health_log.json")

# ── Load health log ───────────────────────────────────────────
def load_log() -> dict:
    try:
        if os.path.exists(LOG_FILE):
            return json.load(open(LOG_FILE))
    except: pass
    return {}

def save_log(log: dict):
    json.dump(log, open(LOG_FILE, "w"), indent=2, default=str)

# ── Individual service testers ────────────────────────────────
def test_groq() -> dict:
    results = []
    for i in range(1, 15):
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if not k: continue
        try:
            from groq import Groq
            Groq(api_key=k).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":"OK"}], max_tokens=3)
            results.append({"key": i, "status": "ok"})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.5)
    return {"service": "groq", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_nvidia() -> dict:
    results = []
    for i in range(1, 15):
        k = os.getenv(f"NVIDIA_API_KEY_{i}")
        if not k: continue
        try:
            from openai import OpenAI
            OpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1"
                ).chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role":"user","content":"OK"}], max_tokens=3)
            results.append({"key": i, "status": "ok"})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.5)
    return {"service": "nvidia", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_openrouter() -> dict:
    results = []
    # Try multiple free models in case one is unavailable
    free_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-2-9b-it:free",
        "qwen/qwen-2-7b-instruct:free",
    ]
    for i in range(1, 15):
        k = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if not k: continue
        worked = False
        for model in free_models:
            try:
                from openai import OpenAI
                OpenAI(api_key=k, base_url="https://openrouter.ai/api/v1"
                    ).chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":"OK"}], max_tokens=3)
                results.append({"key": i, "status": "ok", "model": model})
                worked = True
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    results.append({"key": i, "status": "ok", "note": "rate limited but valid"})
                    worked = True
                    break
                continue
        if not worked:
            results.append({"key": i, "status": "error", "error": "all models failed"})
        time.sleep(0.5)
    return {"service": "openrouter", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_gemini() -> dict:
    results = []
    import google.generativeai as genai
    for i in range(1, 15):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if not k: continue
        try:
            import google.generativeai as genai
            genai.configure(api_key=k)
            genai.GenerativeModel("models/gemini-flash-latest"
                ).generate_content("OK")
            results.append({"key": i, "status": "ok"})
        except Exception as e:
            err = str(e)
            if "429" in err:
                results.append({"key": i, "status": "quota", "error": "rate limited"})
            else:
                results.append({"key": i, "status": "error", "error": err[:50]})
        time.sleep(1)
    return {"service": "gemini", "results": results,
            "working": sum(1 for r in results if r["status"] in ["ok","quota"])}

def test_newsapi() -> dict:
    import requests
    results = []
    for i in range(1, 15):
        k = os.getenv(f"NEWS_API_KEY_{i}")
        if not k: continue
        try:
            r = requests.get("https://newsapi.org/v2/top-headlines",
                params={"apiKey":k,"country":"us","pageSize":1}, timeout=8)
            status = "ok" if r.status_code == 200 else "error"
            results.append({"key": i, "status": status})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.3)
    return {"service": "newsapi", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_gnews() -> dict:
    import requests
    results = []
    for i in range(1, 15):
        k = os.getenv(f"GNEWS_API_KEY_{i}")
        if not k: continue
        try:
            r = requests.get("https://gnews.io/api/v4/top-headlines",
                params={"token":k,"lang":"en","max":1}, timeout=8)
            status = "ok" if r.status_code == 200 else "error"
            results.append({"key": i, "status": status})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.3)
    return {"service": "gnews", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_guardian() -> dict:
    import requests
    results = []
    for i in range(1, 15):
        k = os.getenv(f"GUARDIAN_API_KEY_{i}")
        if not k: continue
        try:
            r = requests.get("https://content.guardianapis.com/search",
                params={"api-key":k,"q":"test","page-size":1}, timeout=8)
            status = "ok" if r.status_code == 200 else "error"
            results.append({"key": i, "status": status})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.3)
    return {"service": "guardian", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_assemblyai() -> dict:
    import requests
    results = []
    for i in range(1, 15):
        k = os.getenv(f"ASSEMBLYAI_API_KEY_{i}")
        if not k: continue
        try:
            r = requests.get("https://api.assemblyai.com/v2/transcript",
                headers={"authorization": k}, timeout=8)
            status = "ok" if r.status_code in [200, 404] else "error"
            results.append({"key": i, "status": status})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.3)
    return {"service": "assemblyai", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_cohere() -> dict:
    results = []
    for i in range(1, 15):
        k = os.getenv(f"COHERE_API_KEY_{i}")
        if not k: continue
        try:
            import cohere
            co = cohere.Client(k)
            co.chat(model="command-r-08-2024", message="OK", max_tokens=10)
            results.append({"key": i, "status": "ok"})
        except Exception as e:
            results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.5)
    return {"service": "cohere", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

def test_xai() -> dict:
    results = []
    for i in range(1, 15):
        k = os.getenv(f"XAI_API_KEY_{i}")
        if not k: continue
        try:
            from openai import OpenAI
            client = OpenAI(api_key=k, base_url="https://api.x.ai/v1")
            client.chat.completions.create(
                model="grok-beta", # Fallback to beta for health check
                messages=[{"role":"user","content":"OK"}], max_tokens=3)
            results.append({"key": i, "status": "ok"})
        except Exception as e:
            err = str(e).lower()
            if "403" in err or "credits" in err or "license" in err or "400" in err:
                results.append({"key": i, "status": "error", "error": "no credits"})
            else:
                results.append({"key": i, "status": "error", "error": str(e)[:50]})
        time.sleep(0.5)
    return {"service": "xai", "results": results,
            "working": sum(1 for r in results if r["status"]=="ok")}

# ── MASTER HEALTH CHECK ───────────────────────────────────────
def run_full_health_check(verbose: bool = True) -> dict:
    """
    Run health check on all API keys.
    Call this periodically to keep keys alive.
    """
    timestamp = datetime.now().isoformat()
    if verbose:
        print("\n" + "="*60)
        print(f"  SAMBHAV KEY HEALTH CHECK — {timestamp[:19]}")
        print("="*60)

    testers = [
        ("Groq",        test_groq),
        ("NVIDIA NIM",  test_nvidia),
        ("OpenRouter",  test_openrouter),
        ("Gemini",      test_gemini),
        ("NewsAPI",     test_newsapi),
        ("GNews",       test_gnews),
        ("Guardian",    test_guardian),
        ("AssemblyAI",  test_assemblyai),
        ("Cohere",      test_cohere),
        ("xAI",         test_xai),
    ]

    report = {"timestamp": timestamp, "services": {}}
    total_working = 0
    total_keys    = 0

    for name, tester in testers:
        if verbose: print(f"\n  Testing {name}...")
        try:
            result = tester()
            working = result["working"]
            total   = len(result["results"])
            total_working += working
            total_keys    += total
            report["services"][name] = result

            if verbose:
                bar = "█" * working + "░" * (total - working)
                status = "✅" if working == total else ("⚠️ " if working > 0 else "❌")
                print(f"    {status} {working}/{total} keys working  [{bar}]")
        except Exception as e:
            if verbose: print(f"    ❌ ERROR: {e}")
            report["services"][name] = {"error": str(e)}

    report["summary"] = {
        "total_working": total_working,
        "total_keys":    total_keys,
        "health_pct":    round(total_working/max(total_keys,1)*100, 1),
        "timestamp":     timestamp,
    }

    if verbose:
        print("\n" + "="*60)
        pct = report["summary"]["health_pct"]
        print(f"  Overall: {total_working}/{total_keys} keys = {pct}% healthy")
        status = "✅ HEALTHY" if pct >= 80 else ("⚠️  DEGRADED" if pct >= 50 else "🔴 CRITICAL")
        print(f"  Status : {status}")
        print("="*60 + "\n")

    # Save log
    log = load_log()
    log[timestamp] = report["summary"]
    # Keep only last 30 entries
    if len(log) > 30:
        oldest = sorted(log.keys())[0]
        del log[oldest]
    save_log(log)

    return report

# ── Scheduler — run every N hours ────────────────────────────
def start_scheduler(interval_hours: int = 24):
    """
    Run health check every N hours to keep keys alive.
    Run as background process.
    """
    print(f"🔄 Key health scheduler started — checking every {interval_hours}h")
    while True:
        run_full_health_check(verbose=False)
        print(f"  ✅ Health check done at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "schedule":
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        start_scheduler(hours)
    else:
        run_full_health_check(verbose=True)
