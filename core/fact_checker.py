"""
core/fact_checker.py — Dual-LLM Fact-Check Module
Section 9 — 8-dimension credibility analysis
Primary: Router (Groq/Llama for structured output)
Secondary: SambaNova for cross-validation
Web search chain: DuckDuckGo -> NewsAPI -> Guardian
"""

import os, logging, time, re
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


# ── Web search chain ──────────────────────────────────────────
def _newsapi_search(query: str) -> list:
    try:
        import requests
        keys = [os.getenv(f"NEWS_API_KEY_{i}") for i in range(1, 9)]
        keys = [k for k in keys if k]
        for key in keys:
            try:
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"q": query, "apiKey": key, "pageSize": 5,
                            "sortBy": "relevancy", "language": "en"},
                    timeout=8)
                if r.status_code == 200:
                    arts = r.json().get("articles", [])
                    return [{"title": a.get("title", ""),
                             "snippet": a.get("description", ""),
                             "link": a.get("url", ""),
                             "source": a.get("source", {}).get("name", "")}
                            for a in arts if a.get("title")]
            except:
                continue
        return []
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
        return []


def _duckduckgo_search(query: str) -> list:
    results = []
    try:
        import requests
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8)
        data = r.json()
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "DuckDuckGo"),
                "snippet": data.get("AbstractText", ""),
                "link": data.get("AbstractURL", ""),
                "source": "DuckDuckGo"})
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": "Related",
                    "snippet": topic["Text"],
                    "link": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo"})
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo failed: {e}")
        return []


def _guardian_search(query: str) -> list:
    try:
        import requests
        keys = [os.getenv(f"GUARDIAN_API_KEY_{i}") for i in range(1, 8)]
        keys = [k for k in keys if k]
        for key in keys:
            try:
                r = requests.get(
                    "https://content.guardianapis.com/search",
                    params={"q": query, "api-key": key,
                            "page-size": 5, "show-fields": "trailText"},
                    timeout=8)
                if r.status_code == 200:
                    items = r.json().get("response", {}).get("results", [])
                    return [{"title": a.get("webTitle", ""),
                             "snippet": a.get("fields", {}).get("trailText", ""),
                             "link": a.get("webUrl", ""),
                             "source": "The Guardian"} for a in items]
            except:
                continue
        return []
    except Exception as e:
        logger.warning(f"Guardian failed: {e}")
        return []


def search_web(query: str) -> list:
    results = _duckduckgo_search(query)
    if len(results) < 2:
        results += _newsapi_search(query)
    if len(results) < 2:
        results += _guardian_search(query)
    logger.info(f"Search '{query[:40]}' -> {len(results)} results")
    return results[:8]


# ── Strip thinking tags ───────────────────────────────────────
def _strip_thinking(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return cleaned.strip()


# ── 8-Dimension Analysis ──────────────────────────────────────
DIMENSIONS = [
    "factual_accuracy", "temporal_accuracy", "geographic_accuracy",
    "source_reliability", "linguistic_precision", "context_completeness",
    "intent_analysis", "viral_risk",
]

SYSTEM_8D = (
    "You are a fact-checking engine for Project Sambhav.\n"
    "Analyze the claim across 8 dimensions. Score each 0-100 (100=fully credible).\n\n"
    "SCORING GUIDE:\n"
    "- factual_accuracy: Is the core claim factually correct?\n"
    "- temporal_accuracy: Was it true when stated? Is it still true?\n"
    "- geographic_accuracy: Is it true globally or only regionally?\n"
    "- source_reliability: Quality of sources supporting this claim\n"
    "- linguistic_precision: Is the claim precisely worded?\n"
    "- context_completeness: Does it include necessary qualifiers?\n"
    "- intent_analysis: Informing vs misleading (100=clearly informing)\n"
    "- viral_risk: How dangerous if widely believed when false (0=safe)\n\n"
    "IMPORTANT: Use your training knowledge. Poor search results do NOT mean false.\n\n"
    "Respond ONLY in this EXACT format:\n"
    "FACTUAL_ACCURACY: <0-100>\n"
    "TEMPORAL_ACCURACY: <0-100>\n"
    "GEOGRAPHIC_ACCURACY: <0-100>\n"
    "SOURCE_RELIABILITY: <0-100>\n"
    "LINGUISTIC_PRECISION: <0-100>\n"
    "CONTEXT_COMPLETENESS: <0-100>\n"
    "INTENT_ANALYSIS: <0-100>\n"
    "VIRAL_RISK: <0-100>\n"
    "OVERALL: <0-100>\n"
    "VERDICT: <VERIFIED_TRUE|LIKELY_TRUE|UNCERTAIN|LIKELY_FALSE|PROBABLY_FALSE|VERIFIED_FALSE>\n"
    "EXPLANATION: <2-3 sentences explaining your verdict>\n"
    "PATTERN_CODE: <ACCURATE|PARTIAL|OUTDATED|MISQUOTE|MISLEADING|FABRICATED|SATIRE>"
)


def analyze_8_dimensions(claim: str, evidence: list) -> dict:
    from llm.router import route

    evidence_text = "\n".join([
        "- [" + r.get("source", "") + "] " + r.get("title", "") + ": " + r.get("snippet", "")[:150]
        for r in evidence[:6]
    ]) or "No external evidence found — rely on training knowledge."

    user_content = (
        "CLAIM: " + claim + "\n\n"
        "WEB EVIDENCE (may be incomplete — also use your training knowledge):\n"
        + evidence_text + "\n\n"
        "Analyze the claim using both evidence and your knowledge. "
        "Do NOT mark false just because search results are poor. "
        "Return ONLY the formatted response."
    )

    messages = [
        {"role": "system", "content": SYSTEM_8D},
        {"role": "user",   "content": user_content},
    ]

    result = route("fact_check", messages, max_tokens=600, temperature=0.1)
    raw = _strip_thinking(result.get("content", ""))
    logger.info(f"Fact-check provider: {result.get('provider_used', 'unknown')}")
    return _parse_8d(raw, claim)


def _parse_8d(raw: str, claim: str) -> dict:
    out = {
        "claim": claim, "raw": raw,
        "factual_accuracy": 50, "temporal_accuracy": 50,
        "geographic_accuracy": 50, "source_reliability": 50,
        "linguistic_precision": 50, "context_completeness": 50,
        "intent_analysis": 50, "viral_risk": 50,
        "overall": 50, "verdict": "UNCERTAIN",
        "explanation": "", "pattern_code": "UNCERTAIN",
    }
    key_map = {
        "FACTUAL_ACCURACY":     "factual_accuracy",
        "TEMPORAL_ACCURACY":    "temporal_accuracy",
        "GEOGRAPHIC_ACCURACY":  "geographic_accuracy",
        "SOURCE_RELIABILITY":   "source_reliability",
        "LINGUISTIC_PRECISION": "linguistic_precision",
        "CONTEXT_COMPLETENESS": "context_completeness",
        "INTENT_ANALYSIS":      "intent_analysis",
        "VIRAL_RISK":           "viral_risk",
        "OVERALL":              "overall",
    }
    for line in raw.split("\n"):
        line = line.strip()
        for key, field in key_map.items():
            if line.startswith(key + ":"):
                try:
                    out[field] = max(0, min(100, int(line.split(":")[1].strip())))
                except:
                    pass
        if line.startswith("VERDICT:"):
            out["verdict"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXPLANATION:"):
            out["explanation"] = line.split(":", 1)[1].strip()
        elif line.startswith("PATTERN_CODE:"):
            out["pattern_code"] = line.split(":", 1)[1].strip()
    return out


# ── Cross-validation with second LLM ─────────────────────────
SYSTEM_CV = (
    "You are an independent fact-checker.\n"
    "Rate the credibility of this claim from 0-100:\n"
    "- 90-100: Verified true, well-documented\n"
    "- 70-89:  Likely true, strong evidence\n"
    "- 50-69:  Uncertain, mixed evidence\n"
    "- 30-49:  Likely false, contradicting evidence\n"
    "- 0-29:   Verified false, definitively wrong\n\n"
    "Respond ONLY in this format:\n"
    "CREDIBILITY: <0-100>\n"
    "REASON: <one sentence based on your knowledge>"
)


def cross_validate(claim: str, primary_score: int) -> dict:
    try:
        from llm.sambanova_client import call_sambanova
        messages = [
            {"role": "system", "content": SYSTEM_CV},
            {"role": "user",   "content": "CLAIM: " + claim},
        ]
        raw = call_sambanova(messages, model="fast", temperature=0.1, max_tokens=100)
        raw = _strip_thinking(raw)
        score = primary_score
        reason = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("CREDIBILITY:"):
                try:
                    score = max(0, min(100, int(line.split(":")[1].strip())))
                except:
                    pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        agreement = abs(score - primary_score) <= 25
        return {
            "secondary_score":   score,
            "reason":            reason,
            "agreement":         agreement,
            "disagreement_flag": not agreement,
            "provider":          "sambanova",
        }
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return {
            "secondary_score":   primary_score,
            "agreement":         True,
            "disagreement_flag": False,
            "provider":          "fallback",
            "reason":            "",
        }


# ── 6 Misinformation Pattern Detectors (Section 9.5) ─────────
def detect_misinformation_patterns(claim: str, analysis: dict) -> list:
    patterns = []
    overall    = analysis.get("overall", 50)
    viral_risk = analysis.get("viral_risk", 0)
    intent     = analysis.get("intent_analysis", 50)
    temporal   = analysis.get("temporal_accuracy", 50)
    factual    = analysis.get("factual_accuracy", 50)
    linguistic = analysis.get("linguistic_precision", 50)
    context    = analysis.get("context_completeness", 50)

    emotional_words = ["hiding", "secret", "truth", "shocking", "exposed",
                       "they don't want", "wake up", "they lied"]
    if any(w in claim.lower() for w in emotional_words) and overall < 60:
        patterns.append({
            "id": "MIS-01", "name": "Emotional Amplification",
            "description": "Uses emotionally charged language to override critical thinking"
        })

    if factual >= 60 and context < 40:
        patterns.append({
            "id": "MIS-02", "name": "Decontextualized Fact",
            "description": "May be technically true but presented without necessary context"
        })

    if factual >= 50 and overall < 50 and analysis.get("source_reliability", 50) < 50:
        patterns.append({
            "id": "MIS-03", "name": "Cherry-Picked Data",
            "description": "Selectively uses data while ignoring contradicting evidence"
        })

    quote_words = ["said", "stated", "claimed", "according to", "declared"]
    if any(w in claim.lower() for w in quote_words) and linguistic < 50:
        patterns.append({
            "id": "MIS-04", "name": "Misattributed Quote",
            "description": "Quote may be inaccurate or misattributed to source"
        })

    if temporal < 40:
        patterns.append({
            "id": "MIS-05", "name": "Outdated Claim",
            "description": "Claim may have been true historically but is no longer current"
        })

    if overall < 30 and viral_risk > 60 and intent < 30:
        patterns.append({
            "id": "MIS-06", "name": "Deliberate Misinformation",
            "description": "Multiple flags suggest intentional false information with high spread risk"
        })

    return patterns


# ── Credibility spectrum ──────────────────────────────────────
CREDIBILITY_SPECTRUM = [
    (90, 100, "VERIFIED",       "Multiple strong sources confirm, no contradictions"),
    (70,  89, "LIKELY_TRUE",    "Strong evidence supports, minor contradictions"),
    (50,  69, "UNCERTAIN",      "Mixed evidence, context-dependent"),
    (30,  49, "LIKELY_FALSE",   "Contradicting evidence stronger than supporting"),
    (10,  29, "PROBABLY_FALSE", "Strong evidence against, very weak support"),
    ( 0,   9, "VERIFIED_FALSE", "Definitively contradicted by authoritative sources"),
]


def get_credibility_label(score: int) -> tuple:
    for lo, hi, label, desc in CREDIBILITY_SPECTRUM:
        if lo <= score <= hi:
            return label, desc
    return "UNCERTAIN", "Unable to determine"


# ── Main fact-check function ──────────────────────────────────
def fact_check_claim(claim: str) -> dict:
    logger.info(f"Fact-checking: {claim[:60]}...")

    evidence = search_web(claim)
    analysis = analyze_8_dimensions(claim, evidence)
    cv = cross_validate(claim, analysis["overall"])

    # Weight primary 80% (has web evidence), secondary 20%
    # Skip secondary if it returned 0 (parse failure)
    sec = cv["secondary_score"]
    if cv["disagreement_flag"] and sec > 5:
        final_score = int(analysis["overall"] * 0.80 + sec * 0.20)
    else:
        final_score = analysis["overall"]

    final_score = max(5, min(99, final_score))
    label, desc = get_credibility_label(final_score)
    patterns = detect_misinformation_patterns(claim, analysis)

    return {
        "claim":                   claim,
        "credibility_score":       final_score,
        "credibility_label":       label,
        "credibility_desc":        desc,
        "dimensions":              {d: analysis.get(d, 50) for d in DIMENSIONS},
        "verdict":                 analysis.get("verdict", "UNCERTAIN"),
        "explanation":             analysis.get("explanation", ""),
        "pattern_code":            analysis.get("pattern_code", "UNCERTAIN"),
        "misinformation_patterns": patterns,
        "cross_validation":        cv,
        "sources":                 evidence[:5],
        "dual_llm_agreement":      not cv["disagreement_flag"],
        "primary_score":           analysis["overall"],
        "secondary_score":         sec,
    }


# ── Batch fact-check ──────────────────────────────────────────
def fact_check_batch(text: str) -> dict:
    import re as _re
    claims = [s.strip() for s in _re.split(r"[.!?]", text)
              if 25 < len(s.strip()) < 300][:12]

    logger.info(f"Batch checking {len(claims)} claims...")
    results = []
    for i, claim in enumerate(claims):
        logger.info(f"  [{i+1}/{len(claims)}] {claim[:50]}...")
        results.append(fact_check_claim(claim))
        time.sleep(0.8)

    scores = [r["credibility_score"] for r in results]
    avg    = int(sum(scores) / len(scores)) if scores else 50
    label, desc = get_credibility_label(avg)

    return {
        "total_claims":        len(results),
        "overall_credibility": avg,
        "overall_label":       label,
        "overall_desc":        desc,
        "claims":              results,
        "verified_count":      sum(1 for r in results if r["credibility_score"] >= 70),
        "false_count":         sum(1 for r in results if r["credibility_score"] < 40),
        "uncertain_count":     sum(1 for r in results if 40 <= r["credibility_score"] < 70),
        "patterns_detected":   list({p["id"] for r in results
                                     for p in r.get("misinformation_patterns", [])}),
    }


if __name__ == "__main__":
    print("Fact-Checker Test\n" + "=" * 50)
    tests = [
        "India became the first country to land on the south pole of the Moon in 2023",
        "The Earth is flat and NASA is hiding the truth",
        "Article 370 was abrogated in India in 2019",
        "India has passed Article 360 in the year 2022",
    ]
    for claim in tests:
        r = fact_check_claim(claim)
        print(f"\nClaim: {claim[:70]}")
        print(f"  Score:      {r['credibility_score']} | {r['credibility_label']}")
        print(f"  Verdict:    {r['verdict']}")
        print(f"  Agreement:  {r['dual_llm_agreement']} (primary={r['primary_score']} secondary={r['secondary_score']})")
        print(f"  Patterns:   {[p['id'] for p in r['misinformation_patterns']]}")
        print(f"  Explanation:{r['explanation'][:120]}")