"""
llm/news_injection.py — Section 8.12 Real-Time News & Context Injection
For predictions involving named organizations, events, or public figures,
fetches live news and injects as additional context signals.
"""

import os, re, logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


def _fetch_news(query: str, max_articles: int = 5) -> list:
    """Fetch news from NewsAPI → GNews → Guardian chain."""
    import requests

    # NewsAPI
    for i in range(1, 9):
        key = os.getenv(f"NEWS_API_KEY_{i}")
        if not key:
            continue
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "apiKey": key, "pageSize": max_articles,
                        "sortBy": "publishedAt", "language": "en"},
                timeout=8)
            if r.status_code == 200:
                arts = r.json().get("articles", [])
                if arts:
                    return [{"title": a.get("title", ""),
                             "description": a.get("description", ""),
                             "source": a.get("source", {}).get("name", ""),
                             "published": a.get("publishedAt", ""),
                             "url": a.get("url", "")}
                            for a in arts if a.get("title")]
        except:
            continue

    # GNews fallback
    for i in range(1, 8):
        key = os.getenv(f"GNEWS_API_KEY_{i}")
        if not key:
            continue
        try:
            r = requests.get(
                f"https://gnews.io/api/v4/search",
                params={"q": query, "token": key, "max": max_articles, "lang": "en"},
                timeout=8)
            if r.status_code == 200:
                arts = r.json().get("articles", [])
                if arts:
                    return [{"title": a.get("title", ""),
                             "description": a.get("description", ""),
                             "source": a.get("source", {}).get("name", ""),
                             "published": a.get("publishedAt", ""),
                             "url": a.get("url", "")}
                            for a in arts if a.get("title")]
        except:
            continue

    return []


def extract_named_entities(text: str) -> list:
    """Extract organization/event names from question for news search."""
    try:
        import spacy
        nlp  = spacy.load("en_core_web_sm")
        doc  = nlp(text)
        entities = [ent.text for ent in doc.ents
                    if ent.label_ in ["ORG", "PERSON", "GPE", "EVENT", "PRODUCT"]]
        return list(set(entities))[:3]
    except:
        # Fallback — extract capitalized words
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(words))[:3]


def inject_news_context(
    domain: str,
    parameters: dict,
    question: str,
    base_probability: float,
) -> dict:
    """
    Section 8.12 — Fetch live news for named entities in the question.
    Returns probability BEFORE and AFTER news injection.
    Shows how current events affect the prediction.
    """
    from llm.router import route
    import json

    # Step 1 — Extract entities to search for
    entities = extract_named_entities(question)
    if not entities:
        return {
            "news_injected": False,
            "reason": "No named entities found in question",
            "probability_before": base_probability,
            "probability_after": base_probability,
            "articles": [],
        }

    # Step 2 — Fetch news
    search_query = " ".join(entities[:2])
    articles = _fetch_news(search_query)

    if not articles:
        return {
            "news_injected": False,
            "reason": f"No news found for: {search_query}",
            "probability_before": base_probability,
            "probability_after": base_probability,
            "articles": [],
            "entities_searched": entities,
        }

    # Step 3 — LLM analyzes news impact
    news_text = "\n".join([
        f"- [{a['source']}] {a['title']}: {a['description'][:100]}"
        for a in articles[:5]
    ])

    param_str = "\n".join([f"  {k}: {v}" for k, v in parameters.items()])

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's news context analyzer.\n"
            "Given a prediction and recent news, assess how the news affects the probability.\n\n"
            "Respond in JSON only:\n"
            "{\n"
            '  "news_sentiment": "<positive|negative|neutral>",\n'
            '  "probability_adjustment": <-0.20 to +0.20>,\n'
            '  "key_news_signal": "<most important news finding>",\n'
            '  "reasoning": "<1-2 sentences explaining the adjustment>",\n'
            '  "news_relevance": "<high|medium|low>"\n'
            "}"
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n"
            f"Base probability: {base_probability*100:.1f}%\n\n"
            f"Recent news about '{search_query}':\n{news_text}\n\n"
            "How does this news affect the prediction probability?"
        )}
    ]

    result = route("llm_predict", messages, max_tokens=300, temperature=0.2)
    raw = result.get("content", "")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        parsed = {
            "news_sentiment": "neutral",
            "probability_adjustment": 0.0,
            "key_news_signal": "Could not parse news impact",
            "reasoning": "News analysis unavailable",
            "news_relevance": "low",
        }

    # Step 4 — Apply adjustment
    adjustment = float(parsed.get("probability_adjustment", 0.0))
    adjustment = max(-0.20, min(0.20, adjustment))
    prob_after = max(0.05, min(0.95, base_probability + adjustment))

    return {
        "news_injected":       True,
        "entities_searched":   entities,
        "search_query":        search_query,
        "probability_before":  round(base_probability, 4),
        "probability_after":   round(prob_after, 4),
        "adjustment":          round(adjustment, 4),
        "adjustment_pct":      f"{adjustment*100:+.1f}%",
        "news_sentiment":      parsed.get("news_sentiment", "neutral"),
        "key_news_signal":     parsed.get("key_news_signal", ""),
        "reasoning":           parsed.get("reasoning", ""),
        "news_relevance":      parsed.get("news_relevance", "low"),
        "articles":            articles[:5],
        "provider":            result.get("provider_used", "unknown"),
    }


if __name__ == "__main__":
    print("News Injection Test\n" + "="*40)
    result = inject_news_context(
        domain="hr",
        parameters={"job_satisfaction": 2, "overtime": 1, "years_at_company": 3},
        question="Will the employee at Google resign this year?",
        base_probability=0.65,
    )
    print(f"News injected:     {result['news_injected']}")
    print(f"Entities searched: {result.get('entities_searched', [])}")
    print(f"Probability before:{result['probability_before']*100:.1f}%")
    print(f"Probability after: {result['probability_after']*100:.1f}%")
    print(f"Adjustment:        {result.get('adjustment_pct', '0%')}")
    print(f"Key signal:        {result.get('key_news_signal', '')}")
    print(f"Reasoning:         {result.get('reasoning', '')}")
    print(f"Articles found:    {len(result.get('articles', []))}")
