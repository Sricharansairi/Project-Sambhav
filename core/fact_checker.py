import os, logging, time
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Web search chain ──────────────────────────────────────────
def _google_search(query: str, num: int = 5) -> list:
    try:
        import requests
        keys = [os.getenv(f"GOOGLE_SEARCH_API_KEY_{i}") for i in range(1,11)]
        keys = [k for k in keys if k]
        cx   = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        for key in keys:
            try:
                r = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"key":key,"cx":cx,"q":query,"num":num},
                    timeout=8)
                if r.status_code == 200:
                    items = r.json().get("items", [])
                    return [{"title":i.get("title",""),
                             "snippet":i.get("snippet",""),
                             "link":i.get("link","")} for i in items]
            except: continue
        return []
    except Exception as e:
        logger.warning(f"Google search failed: {e}")
        return []

def _newsapi_search(query: str) -> list:
    try:
        import requests
        keys = [os.getenv(f"NEWS_API_KEY_{i}") for i in range(1,5)]
        keys = [k for k in keys if k]
        for key in keys:
            try:
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"q":query,"apiKey":key,"pageSize":5,
                            "sortBy":"relevancy","language":"en"},
                    timeout=8)
                if r.status_code == 200:
                    arts = r.json().get("articles",[])
                    return [{"title":a.get("title",""),
                             "snippet":a.get("description",""),
                             "link":a.get("url",""),
                             "source":a.get("source",{}).get("name","")}
                            for a in arts]
            except: continue
        return []
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
        return []

def _duckduckgo_search(query: str) -> list:
    """DuckDuckGo instant answer + HTML scrape fallback — no API key needed."""
    results = []
    try:
        import requests
        # Try instant answer API
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q":query,"format":"json","no_html":1,"skip_disambig":1},
            headers={"User-Agent":"Mozilla/5.0"},
            timeout=8)
        data = r.json()
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading","DuckDuckGo"),
                "snippet": data.get("AbstractText",""),
                "link":    data.get("AbstractURL",""),
                "source":  "DuckDuckGo"})
        for topic in data.get("RelatedTopics",[])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   "Related",
                    "snippet": topic["Text"],
                    "link":    topic.get("FirstURL",""),
                    "source":  "DuckDuckGo"})
        if results:
            return results
    except Exception as e:
        logger.warning(f"DuckDuckGo instant failed: {e}")

    # Fallback — use duckduckgo_search library if installed
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title":   r.get("title",""),
                    "snippet": r.get("body",""),
                    "link":    r.get("href",""),
                    "source":  "DuckDuckGo"})
        return results
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"DDGS fallback failed: {e}")

    # Final fallback — use NewsAPI
    return _newsapi_search(query)

def _guardian_search(query: str) -> list:
    """The Guardian API — free, no quota issues."""
    try:
        import requests
        from api.key_rotator import get_key
        key = get_key("guardian")
        r   = requests.get(
            "https://content.guardianapis.com/search",
            params={"q": query, "api-key": key,
                    "page-size": 5, "show-fields": "trailText"},
            timeout=8)
        if r.status_code == 200:
            results = r.json().get("response",{}).get("results",[])
            return [{"title":   a.get("webTitle",""),
                     "snippet": a.get("fields",{}).get("trailText",""),
                     "link":    a.get("webUrl",""),
                     "source":  "The Guardian"}
                    for a in results]
        return []
    except Exception as e:
        logger.warning(f"Guardian search failed: {e}")
        return []

def search_web(query: str) -> list:
    """Search chain: DuckDuckGo → NewsAPI → Guardian → GNews."""
    results = _duckduckgo_search(query)
    if not results:
        results = _newsapi_search(query)
    if not results:
        results = _guardian_search(query)
    logger.info(f"Search '{query[:40]}' → {len(results)} results")
    return results

# ── 8-Dimension Analysis ──────────────────────────────────────
DIMENSIONS = [
    "factual_accuracy",
    "temporal_accuracy",
    "geographic_accuracy",
    "source_reliability",
    "linguistic_precision",
    "context_completeness",
    "intent_analysis",
    "viral_risk",
]

def analyze_8_dimensions(claim: str, evidence: list) -> dict:
    """Run 8-dimension analysis via dual LLM."""
    from llm.groq_client import call_groq

    evidence_text = "\n".join([
        f"- {r.get('title','')}: {r.get('snippet','')}"
        for r in evidence[:6]
    ]) or "No external evidence found."

    messages = [
        {"role": "system", "content": (
            "You are a fact-checking engine. Analyze the claim across 8 dimensions. "
            "For each dimension score 0-100 (100=perfect). "
            "Respond ONLY in this exact format:\n"
            "FACTUAL_ACCURACY: <0-100>\n"
            "TEMPORAL_ACCURACY: <0-100>\n"
            "GEOGRAPHIC_ACCURACY: <0-100>\n"
            "SOURCE_RELIABILITY: <0-100>\n"
            "LINGUISTIC_PRECISION: <0-100>\n"
            "CONTEXT_COMPLETENESS: <0-100>\n"
            "INTENT_ANALYSIS: <0-100>\n"
            "VIRAL_RISK: <0-100>\n"
            "OVERALL: <0-100>\n"
            "VERDICT: <VERIFIED|LIKELY_TRUE|UNCERTAIN|LIKELY_FALSE|PROBABLY_FALSE|VERIFIED_FALSE>\n"
            "EXPLANATION: <2-3 sentences>\n"
            "PATTERN_CODE: <MISQUOTE|OUTDATED|MISLEADING|SATIRE|FABRICATED|ACCURATE|PARTIAL>"
        )},
        {"role": "user", "content": (
            f"Claim: {claim}\n\n"
            f"Evidence found:\n{evidence_text}\n\n"
            "Analyze across all 8 dimensions."
        )}
    ]

    raw = call_groq(messages, temperature=0.1, max_tokens=500)
    return _parse_8d_response(raw, claim)

def _parse_8d_response(raw: str, claim: str) -> dict:
    result = {
        "claim":              claim,
        "raw":                raw,
        "factual_accuracy":   50,
        "temporal_accuracy":  50,
        "geographic_accuracy":50,
        "source_reliability": 50,
        "linguistic_precision":50,
        "context_completeness":50,
        "intent_analysis":    50,
        "viral_risk":         50,
        "overall":            50,
        "verdict":            "UNCERTAIN",
        "explanation":        "",
        "pattern_code":       "UNCERTAIN",
    }
    for line in raw.split("\n"):
        line = line.strip()
        if   line.startswith("FACTUAL_ACCURACY:"):
            try: result["factual_accuracy"]    = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("TEMPORAL_ACCURACY:"):
            try: result["temporal_accuracy"]   = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("GEOGRAPHIC_ACCURACY:"):
            try: result["geographic_accuracy"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("SOURCE_RELIABILITY:"):
            try: result["source_reliability"]  = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("LINGUISTIC_PRECISION:"):
            try: result["linguistic_precision"]= int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("CONTEXT_COMPLETENESS:"):
            try: result["context_completeness"]= int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("INTENT_ANALYSIS:"):
            try: result["intent_analysis"]     = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("VIRAL_RISK:"):
            try: result["viral_risk"]          = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("OVERALL:"):
            try:
                val = line.split(":")[1].strip()
                result["overall"] = max(1, int(val))  # never let score be 0
            except: pass
        elif line.startswith("VERDICT:"):
            result["verdict"]      = line.split(":",1)[1].strip()
        elif line.startswith("EXPLANATION:"):
            result["explanation"]  = line.split(":",1)[1].strip()
        elif line.startswith("PATTERN_CODE:"):
            result["pattern_code"] = line.split(":",1)[1].strip()
    return result

# ── Credibility spectrum ──────────────────────────────────────
CREDIBILITY_SPECTRUM = [
    (90, 100, "VERIFIED",        "Multiple strong sources, no contradictions"),
    (70,  89, "LIKELY_TRUE",     "Strong evidence, minor contradictions"),
    (50,  69, "UNCERTAIN",       "Mixed evidence, conflicting sources"),
    (30,  49, "LIKELY_FALSE",    "Contradicting evidence stronger"),
    (10,  29, "PROBABLY_FALSE",  "Strong evidence against"),
    ( 0,   9, "VERIFIED_FALSE",  "Definitively contradicted"),
]

def get_credibility_label(score: int) -> tuple:
    for lo, hi, label, desc in CREDIBILITY_SPECTRUM:
        if lo <= score <= hi:
            return label, desc
    return "UNCERTAIN", "Unable to determine"

# ── Cross-validation with second LLM ─────────────────────────
def cross_validate(claim: str, primary_score: int) -> dict:
    """Second LLM independently validates — flags if disagreement > 20."""
    try:
        from llm.groq_client import call_groq
        messages = [
            {"role": "system", "content": (
                "You are an independent fact-checker. "
                "Rate the credibility of this claim from 0-100. "
                "Respond ONLY: CREDIBILITY: <0-100>\nREASON: <1 sentence>"
            )},
            {"role": "user", "content": f"Claim: {claim}"}
        ]
        # Use different temperature for independence
        raw   = call_groq(messages, temperature=0.4, max_tokens=100)
        score = 50
        reason= ""
        for line in raw.split("\n"):
            if line.startswith("CREDIBILITY:"):
                try: score = int(line.split(":")[1].strip())
                except: pass
            elif line.startswith("REASON:"):
                reason = line.split(":",1)[1].strip()

        agreement = abs(score - primary_score) <= 20
        return {"secondary_score": score, "reason": reason,
                "agreement": agreement,
                "disagreement_flag": not agreement}
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return {"secondary_score": primary_score, "agreement": True,
                "disagreement_flag": False}

# ── Single claim fact-check ───────────────────────────────────
def fact_check_claim(claim: str) -> dict:
    """
    Full fact-check pipeline for a single claim:
    1. Web search for evidence
    2. 8-dimension analysis
    3. Cross-validate with second LLM
    4. Return credibility score + full report
    """
    logger.info(f"Fact-checking: {claim[:60]}...")

    # Step 1 — Search
    evidence = search_web(claim)
    time.sleep(0.5)  # rate limit protection

    # Step 2 — 8D analysis
    analysis = analyze_8_dimensions(claim, evidence)

    # Step 3 — Cross-validate
    cv = cross_validate(claim, analysis["overall"])

    # Step 4 — Final score (weighted average)
    if cv["disagreement_flag"]:
        # Average when disagreement — show uncertainty
        final_score = int((analysis["overall"] + cv["secondary_score"]) / 2)
    else:
        final_score = analysis["overall"]

    label, desc = get_credibility_label(final_score)

    return {
        "claim":              claim,
        "credibility_score":  final_score,
        "credibility_label":  label,
        "credibility_desc":   desc,
        "dimensions":         {d: analysis.get(d, 50) for d in DIMENSIONS},
        "verdict":            analysis.get("verdict", "UNCERTAIN"),
        "explanation":        analysis.get("explanation", ""),
        "pattern_code":       analysis.get("pattern_code", "UNCERTAIN"),
        "cross_validation":   cv,
        "sources":            evidence[:5],
        "dual_llm_agreement": not cv["disagreement_flag"],
    }

# ── Batch fact-check ──────────────────────────────────────────
def fact_check_batch(text: str) -> dict:
    """
    Fact-check an entire article/speech.
    Extracts all claims → checks each → returns annotated report.
    """
    from vision.document_pipeline import extract_claims
    claims  = extract_claims(text)
    if not claims:
        # Fallback — split by sentences
        import re
        claims = [s.strip() for s in re.split(r'[.!?]', text)
                  if len(s.strip()) > 20][:10]

    logger.info(f"Batch checking {len(claims)} claims...")
    results = []
    for i, claim in enumerate(claims[:15]):  # cap at 15
        logger.info(f"  Claim {i+1}/{len(claims)}: {claim[:40]}...")
        result = fact_check_claim(claim)
        results.append(result)
        time.sleep(1.0)  # rate limit protection

    # Overall article credibility
    scores  = [r["credibility_score"] for r in results]
    avg     = int(sum(scores)/len(scores)) if scores else 50
    label, desc = get_credibility_label(avg)

    return {
        "total_claims":          len(results),
        "overall_credibility":   avg,
        "overall_label":         label,
        "overall_desc":          desc,
        "claims":                results,
        "verified_count":        sum(1 for r in results if r["credibility_score"] >= 70),
        "false_count":           sum(1 for r in results if r["credibility_score"] < 40),
        "uncertain_count":       sum(1 for r in results if 40 <= r["credibility_score"] < 70),
    }

if __name__ == "__main__":
    print("\n🔍 Testing Fact-Checker...\n")
    claim  = "India became the first country to land on the south pole of the Moon in 2023"
    result = fact_check_claim(claim)
    print(f"  Claim      : {claim}")
    print(f"  Score      : {result['credibility_score']}")
    print(f"  Label      : {result['credibility_label']}")
    print(f"  Verdict    : {result['verdict']}")
    print(f"  Explanation: {result['explanation']}")
    print(f"  Agreement  : {result['dual_llm_agreement']}")
    print(f"  Sources    : {len(result['sources'])}")
    print(f"  Dimensions : {result['dimensions']}")
