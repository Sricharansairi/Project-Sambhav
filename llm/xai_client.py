import os, sys, time, logging
# Get project root and add to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

def fact_check_with_grok(claim: str) -> dict:
    """
    Real-time fact check.
    Primary: xAI Grok (if credits available)
    Fallback: Groq + DuckDuckGo search
    """
    # Try xAI first
    try:
        from openai import OpenAI
        keys = [os.getenv(f"XAI_API_KEY_{i}") for i in range(1,15)]
        keys = [k for k in keys if k]
        if keys:
            client = OpenAI(api_key=keys[0], base_url="https://api.x.ai/v1")
            resp = client.chat.completions.create(
                model="grok-2-1212",
                messages=[
                    {"role":"system","content":(
                        "You are a real-time fact checker. "
                        "Respond ONLY in this format:\n"
                        "CREDIBILITY: <0-100>\n"
                        "VERDICT: <VERIFIED|LIKELY_TRUE|UNCERTAIN|LIKELY_FALSE|VERIFIED_FALSE>\n"
                        "EXPLANATION: <2-3 sentences>\n"
                        "SOURCES: <sources used>"
                    )},
                    {"role":"user","content":f"Fact-check: {claim}"}
                ], max_tokens=200, timeout=10)
            raw = resp.choices[0].message.content
            return _parse_response(raw)
    except Exception as e:
        if "403" in str(e) or "credits" in str(e).lower():
            logger.info("xAI no credits — using Groq fallback")
        else:
            logger.warning(f"xAI failed: {e}")

    # Fallback — Groq + web search
    return _groq_fact_check(claim)

def _groq_fact_check(claim: str) -> dict:
    """Groq-based fact check with DuckDuckGo search context."""
    try:
        from llm.groq_client import call_groq
        from core.fact_checker import search_web

        # Get web evidence first
        evidence = search_web(claim)
        evidence_text = "\n".join([
            f"- {r.get('title','')}: {r.get('snippet','')}"
            for r in evidence[:5]
        ]) or "No web evidence found."

        messages = [
            {"role":"system","content":(
                "You are a fact checker with web search results provided. "
                "Analyze the claim against the evidence. "
                "Respond ONLY in this format:\n"
                "CREDIBILITY: <0-100>\n"
                "VERDICT: <VERIFIED|LIKELY_TRUE|UNCERTAIN|LIKELY_FALSE|VERIFIED_FALSE>\n"
                "EXPLANATION: <2-3 sentences>\n"
                "SOURCES: <sources used>"
            )},
            {"role":"user","content":(
                f"Claim: {claim}\n\n"
                f"Web evidence:\n{evidence_text}"
            )}
        ]
        raw = call_groq(messages, temperature=0.1, max_tokens=200)
        return _parse_response(raw)
    except Exception as e:
        logger.error(f"Groq fact check failed: {e}")
        return {"credibility": 50, "verdict": "UNCERTAIN",
                "explanation": "Unable to verify claim",
                "sources": [], "raw": ""}

def _parse_response(raw: str) -> dict:
    result = {"raw": raw, "credibility": 50,
              "verdict": "UNCERTAIN", "explanation": "",
              "sources": []}
    for line in raw.split("\n"):
        if line.startswith("CREDIBILITY:"):
            try: result["credibility"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("VERDICT:"):
            result["verdict"] = line.split(":",1)[1].strip()
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.split(":",1)[1].strip()
        elif line.startswith("SOURCES:"):
            result["sources"] = [s.strip() for s in line.split(":",1)[1].split(",")]
    return result

def health_check() -> dict:
    try:
        r = fact_check_with_grok("The Earth orbits the Sun")
        return {"status":"ok","provider":"groq_fallback",
                "keys":8,"credibility":r.get("credibility")}
    except Exception as e:
        return {"status":"error","error":str(e)}

if __name__ == "__main__":
    print("Testing xAI/Groq fact check...")
    r = fact_check_with_grok("India landed on the moon's south pole in 2023")
    print(f"Credibility: {r['credibility']}")
    print(f"Verdict: {r['verdict']}")
    print(f"Explanation: {r['explanation']}")
