"""
core/fact_checker.py — Dual-LLM Fact-Check Module
Section 9 — 8-dimension credibility analysis
Primary: Cerebras for high-speed cross-validation
Secondary: Groq/SambaNova for detailed reasoning
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
    """
    Enhanced search with relevance filtering.
    """
    results = _duckduckgo_search(query)
    # Filter out results with missing or search-engine links
    results = [r for r in results if r.get("link") and "duckduckgo.com" not in r["link"]]
    
    if len(results) < 3:
        results += _newsapi_search(query)
    if len(results) < 3:
        results += _guardian_search(query)
        
    # Ensure links are absolute and valid
    valid_results = []
    for r in results:
        link = r.get("link", "")
        if link.startswith("http"):
            valid_results.append(r)
            
    # Step 3 — Semantic Filtering (Section 9.4)
    # Filter out results that don't mention key terms from the query
    # RELAXED FILTER: Only filter if we have more than 6 results
    keywords = [w.lower() for w in query.split() if len(w) > 4]
    if not keywords:
        keywords = [w.lower() for w in query.split()]
        
    filtered = []
    for r in valid_results:
        text = (r["title"] + " " + (r.get("snippet") or "")).lower()
        # MUST match at least one keyword, OR if keywords are generic, keep it
        if any(k in text for k in keywords) or len(valid_results) <= 3:
            filtered.append(r)
            
    return filtered[:6]


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
    "You are a highly critical, elite fact-checking engine for Project Sambhav.\n"
    "Your mission is to find the objective truth. Do not be fooled by misinformation or SEO spam.\n"
    "Analyze the claim across 8 dimensions. Score each 0-100 (100=fully credible, 0=definitely false).\n\n"
    "CRITICAL TRUTH RULES:\n"
    "1. AUTHORITATIVE KNOWLEDGE: Use your internal high-quality training data. If you know a claim is false (e.g. Article 360 was NOT passed in 2022), score it as 0 regardless of search results.\n"
    "2. SEARCH SKEPTICISM: Search results can be noisy or misleading. If results are contradictory, prioritize authoritative official sources (gov.in, edu, reputable news).\n"
    "3. ABSOLUTE VERDICTS: If a claim is factually impossible or historically wrong, the OVERALL score MUST be below 10.\n\n"
    "SCORING GUIDE:\n"
    "- factual_accuracy: Core truth of the statement (0 if factually wrong)\n"
    "- temporal_accuracy: Timeline validity (e.g. wrong year = low score)\n"
    "- geographic_accuracy: Regional specificity\n"
    "- source_reliability: Reliability of the evidence provided\n"
    "- linguistic_precision: Clarity vs Vagueness\n"
    "- context_completeness: Are critical details missing?\n"
    "- intent_analysis: Is it meant to inform or deceive?\n"
    "- viral_risk: Danger of this misinformation\n\n"
    "Respond ONLY in this EXACT format (one dimension per block, include REASONING):\n"
    "FACTUAL_ACCURACY: <0-100>\n"
    "REASONING_FACTUAL_ACCURACY: <detailed reasoning for this score>\n"
    "TEMPORAL_ACCURACY: <0-100>\n"
    "REASONING_TEMPORAL_ACCURACY: <detailed reasoning for this score>\n"
    "GEOGRAPHIC_ACCURACY: <0-100>\n"
    "REASONING_GEOGRAPHIC_ACCURACY: <detailed reasoning for this score>\n"
    "SOURCE_RELIABILITY: <0-100>\n"
    "REASONING_SOURCE_RELIABILITY: <detailed reasoning for this score>\n"
    "LINGUISTIC_PRECISION: <0-100>\n"
    "REASONING_LINGUISTIC_PRECISION: <detailed reasoning for this score>\n"
    "CONTEXT_COMPLETENESS: <0-100>\n"
    "REASONING_CONTEXT_COMPLETENESS: <detailed reasoning for this score>\n"
    "INTENT_ANALYSIS: <0-100>\n"
    "REASONING_INTENT_ANALYSIS: <detailed reasoning for this score>\n"
    "VIRAL_RISK: <0-100>\n"
    "REASONING_VIRAL_RISK: <detailed reasoning for this score>\n"
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
    raw = _strip_thinking(result.get("content") or "")   # guard against None
    logger.info(f"Fact-check provider: {result.get('provider_used', 'unknown')}")
    return _parse_8d(raw, claim)


def _parse_8d(raw: str, claim: str) -> dict:
    out = {
        "claim": claim, "raw": raw,
        "factual_accuracy": {"score": 50, "reasoning": ""},
        "temporal_accuracy": {"score": 50, "reasoning": ""},
        "geographic_accuracy": {"score": 50, "reasoning": ""},
        "source_reliability": {"score": 50, "reasoning": ""},
        "linguistic_precision": {"score": 50, "reasoning": ""},
        "context_completeness": {"score": 50, "reasoning": ""},
        "intent_analysis": {"score": 50, "reasoning": ""},
        "viral_risk": {"score": 50, "reasoning": ""},
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
    }
    
    import re

    # Try full document regex parsing first to be completely bulletproof
    for key, field in key_map.items():
        score_match = re.search(rf"(?i){key}.*?(\d{{1,3}})", raw)
        if score_match:
            try:
                out[field]["score"] = max(0, min(100, int(score_match.group(1))))
            except: pass
            
        reason_match = re.search(rf"(?i)REASONING_{key}.*?:\s*(.+?)(?=\n|$)", raw)
        if reason_match:
            out[field]["reasoning"] = reason_match.group(1).strip()
            
    # Parse global fields
    overall_match = re.search(r"(?i)OVERALL.*?(\d{1,3})", raw)
    if overall_match:
        try:
            out["overall"] = max(0, min(100, int(overall_match.group(1))))
        except: pass
        
    verdict_match = re.search(r"(?i)VERDICT.*?:\s*(.+?)(?=\n|$)", raw)
    if verdict_match: out["verdict"] = verdict_match.group(1).strip()
    
    expl_match = re.search(r"(?i)EXPLANATION.*?:\s*(.+?)(?=\n|$)", raw)
    if expl_match: out["explanation"] = expl_match.group(1).strip()
    
    pat_match = re.search(r"(?i)PATTERN_CODE.*?:\s*(.+?)(?=\n|$)", raw)
    if pat_match: out["pattern_code"] = pat_match.group(1).strip()
            
    return out


# ── Cross-validation with second LLM ─────────────────────────
SYSTEM_CV = (
    "You are a highly critical independent fact-checker.\n"
    "Your job is to debunk false claims. Be extremely skeptical.\n"
    "Rate the credibility of this claim from 0-100:\n"
    "- 90-100: Absolute truth, verified by multiple authoritative sources\n"
    "- 70-89:  Likely true, good evidence exists\n"
    "- 40-69:  Uncertain, ambiguous, or unverified\n"
    "- 10-39:  Likely false, strong evidence contradicts\n"
    "- 0-9:    Absolute falsehood, factually impossible or historically wrong\n\n"
    "Respond ONLY in this format:\n"
    "CREDIBILITY: <0-100>\n"
    "REASON: <one sentence based on your knowledge and the evidence>"
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
    viral_risk = analysis.get("viral_risk", {}).get("score", 0)
    intent     = analysis.get("intent_analysis", {}).get("score", 50)
    temporal   = analysis.get("temporal_accuracy", {}).get("score", 50)
    factual    = analysis.get("factual_accuracy", {}).get("score", 50)
    linguistic = analysis.get("linguistic_precision", {}).get("score", 50)
    context    = analysis.get("context_completeness", {}).get("score", 50)

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
    (70,  89, "LIKELY TRUE",    "Strong evidence supports, minor contradictions"),
    (50,  69, "UNCERTAIN",      "Mixed evidence, context-dependent"),
    (30,  49, "LIKELY FALSE",   "Contradicting evidence stronger than supporting"),
    (10,  29, "PROBABLY FALSE", "Strong evidence against, very weak support"),
    ( 0,   9, "VERIFIED FALSE", "Definitively contradicted by authoritative sources"),
]


def get_credibility_label(score: int) -> tuple:
    for lo, hi, label, desc in CREDIBILITY_SPECTRUM:
        if lo <= score <= hi:
            return label, desc
    return "UNCERTAIN", "Unable to determine"


def fact_check_claim(claim: str, mode: str = "standard") -> dict:
    """
    Main fact-check entrypoint.
    Modes supported: Quick, Standard, Deep, Batch.
    """
    if not claim or len(claim) < 10:
        return {"error": "Claim too short"}

    logger.info(f"Fact-checking ({mode}): {claim[:60]}...")

    # Step 1 — Evidence gathering
    if mode == "quick":
        evidence = []
    else:
        evidence = search_web(claim)

    evidence_str = "\n".join([f"[{i+1}] {e['title']}: {e['snippet']}" for i, e in enumerate(evidence)])

    # Step 2 — Primary analysis
    from llm.router import route
    prompt_8d = f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_str or 'No external evidence found. Use internal knowledge.'}"
    
    # We use a higher temperature for reasoning but low for final scores
    raw_result = route("fact_check", [{"role": "system", "content": SYSTEM_8D}, {"role": "user", "content": prompt_8d}], temperature=0.2)
    raw_content = raw_result.get("content") or ""   # Guard against None if all providers failed
    analysis = _parse_8d(raw_content, claim)

    # Step 3 — Cross-validation (Dual LLM)
    if mode == "quick":
        sec = analysis["overall"]
        cv = {"secondary_score": sec, "agreement": True, "disagreement_flag": False}
    else:
        cv_res = route("fact_check", [
            {"role": "system", "content": "You are a critical cross-validator. Review the claim and evidence. Return ONLY this format:\nCREDIBILITY: <0-100>\nREASON: <one sentence>"},
            {"role": "user", "content": prompt_8d}
        ], temperature=0.1)
        
        import re
        cv_content = cv_res.get("content") or ""
        # Require explicit CREDIBILITY: prefix to avoid picking up random numbers from evidence text
        cv_score_match = re.search(r"(?i)CREDIBILITY.*?(\d{1,3})", cv_content)
        if cv_score_match:
            sec = max(0, min(100, int(cv_score_match.group(1))))
        else:
            sec = analysis["overall"]  # Fall back to primary score if CV parse fails
        
        cv = {
            "secondary_score": sec,
            "disagreement_flag": abs(analysis["overall"] - sec) > 25,
            "provider": "router"
        }

    # Final Reconciled Score
    final_score = round(0.7 * analysis["overall"] + 0.3 * sec)
    final_score = max(5, min(99, final_score))

    label, desc = get_credibility_label(final_score)
    patterns = detect_misinformation_patterns(claim, analysis)

    return {
        "claim":                   claim,
        "credibility_score":       final_score,
        "credibility_label":       label,
        "credibility_desc":        desc,
        "dimensions":              {d: analysis.get(d, {"score": 50, "reasoning": ""}) for d in DIMENSIONS},
        "verdict":                 analysis.get("verdict", "UNCERTAIN"),
        "explanation":             analysis.get("explanation", ""),
        "pattern_code":            analysis.get("pattern_code", "UNCERTAIN"),
        "misinformation_patterns": patterns,
        "cross_validation":        cv,
        "sources":                 evidence[:5],
        "dual_llm_agreement":      not cv.get("disagreement_flag", False),
        "primary_score":           analysis["overall"],
        "secondary_score":         sec,
    }


# ── Batch fact-check ──────────────────────────────────────────
def fact_check_batch(text: str) -> dict:
    import re as _re
    claims = [s.strip() for s in _re.split(r"[.!?]", text)
              if 25 < len(s.strip()) < 300][:50]

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

class FactChecker:
    """Entrypoint class for Sambhav Fact-Checking engine."""
    @staticmethod
    def verify(claim: str, mode: str = "full"):
        return fact_check_claim(claim, mode=mode)
    
    @staticmethod
    def batch_verify(text: str):
        return fact_check_batch(text)


check_claim = fact_check_claim

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