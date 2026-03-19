"""
llm/router.py — Master LLM Router for Project Sambhav
Routes all LLM calls through correct provider chain per function type.
Based on Section 12 (Multi-API Architecture) + Section 5.4 (LLM Layer)
Four-level failover cascade per Section 12.4
"""

import logging
from typing import Optional
from llm.groq_client import call_groq
from llm.sambanova_client import call_sambanova
from llm.cerebras_client import call_cerebras
from llm.mistral_client import call_mistral
from llm.cloudflare_client import call_cloudflare

logger = logging.getLogger(__name__)

# ── Temperature map per call type ─────────────────────────────
TEMPERATURES = {
    "domain_detect":      0.1,
    "llm_predict":        0.2,
    "fact_check":         0.1,
    "free_inference":     0.5,
    "outcome_simulation": 0.7,
    "multi_agent_debate": 0.4,
    "devils_advocate":    0.3,
    "conversational":     0.3,
    "safety_screen":      0.1,
    "document_analysis":  0.2,
}

# ── Provider chains per call type (Section 12.4) ─────────────
CHAINS = {
    "domain_detect": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "fast"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "llm_predict": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "fast"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "free_inference": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "large"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "fact_check": [
        ("cerebras",    {"model": "large"}),
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("mistral",     {"model": "large"}),
        ("sambanova",   {"model": "fast"}),
    ],
    "multi_agent_debate": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "fast"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "devils_advocate": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "reasoning"}),
        ("mistral",     {"model": "large"}),
        ("cerebras",    {"model": "large"}),
    ],
    "conversational": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "fast"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "document_analysis": [
        ("sambanova",   {"model": "document"}),
        ("mistral",     {"model": "large"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "outcome_simulation": [
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "large"}),
        ("cerebras",    {"model": "large"}),
        ("cloudflare",  {"model": "large"}),
    ],
    "safety_screen": [
        ("mistral",     {"model": "small"}),
        ("groq",        {"model": "llama-3.3-70b-versatile"}),
        ("sambanova",   {"model": "fast"}),
        ("cerebras",    {"model": "large"}),
    ],
}

# ── Provider caller map ───────────────────────────────────────
def _call_provider(provider: str, messages: list, model_kwargs: dict,
                   temperature: float, max_tokens: int) -> str:
    if provider == "groq":
        return call_groq(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_kwargs.get("model", "llama-3.3-70b-versatile")
        )
    elif provider == "sambanova":
        return call_sambanova(
            messages,
            model=model_kwargs.get("model", "fast"),
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "cerebras":
        return call_cerebras(
            messages,
            model=model_kwargs.get("model", "large"),
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "mistral":
        return call_mistral(
            messages,
            model=model_kwargs.get("model", "large"),
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "cloudflare":
        return call_cloudflare(
            messages,
            model=model_kwargs.get("model", "large"),
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Master router function ────────────────────────────────────
def route(
    call_type: str,
    messages: list,
    max_tokens: int = 1000,
    temperature: Optional[float] = None,
    force_provider: Optional[str] = None,
) -> dict:
    """
    Route an LLM call through the correct provider chain.

    Args:
        call_type: One of CHAINS keys — determines provider order
        messages: List of {role, content} dicts
        max_tokens: Max tokens for response
        temperature: Override default temperature if needed
        force_provider: Skip chain and use specific provider

    Returns:
        dict with keys: content, provider_used, call_type, success
    """
    temp = temperature or TEMPERATURES.get(call_type, 0.2)
    chain = CHAINS.get(call_type, CHAINS["llm_predict"])

    # Force specific provider if requested
    if force_provider:
        chain = [(p, k) for p, k in chain if p == force_provider] or chain

    last_error = None
    for level, (provider, model_kwargs) in enumerate(chain):
        try:
            logger.info(f"[Router] {call_type} → {provider} (level {level+1})")
            content = _call_provider(
                provider, messages, model_kwargs, temp, max_tokens
            )
            return {
                "content":       content,
                "provider_used": provider,
                "call_type":     call_type,
                "level":         level + 1,
                "success":       True,
            }
        except Exception as e:
            last_error = e
            logger.warning(
                f"[Router] {provider} failed for {call_type} "
                f"(level {level+1}): {e} — trying next provider"
            )
            continue

    # Level 4 — all providers failed
    logger.error(f"[Router] ALL providers failed for {call_type}: {last_error}")
    return {
        "content":       None,
        "provider_used": None,
        "call_type":     call_type,
        "level":         None,
        "success":       False,
        "error":         str(last_error),
    }


# ── Health check all providers ────────────────────────────────
def health_check_all() -> dict:
    results = {}
    test_messages = [{"role": "user", "content": "Reply OK"}]
    providers = [
        ("groq",       lambda: call_groq(test_messages, max_tokens=5)),
        ("sambanova",  lambda: call_sambanova(test_messages, max_tokens=5)),
        ("cerebras",   lambda: call_cerebras(test_messages, max_tokens=5)),
        ("mistral",    lambda: call_mistral(test_messages, max_tokens=5)),
        ("cloudflare", lambda: call_cloudflare(test_messages, max_tokens=5)),
    ]
    for name, fn in providers:
        try:
            fn()
            results[name] = "GOOD"
        except Exception as e:
            results[name] = f"FAILED — {str(e)[:50]}"
    return results


if __name__ == "__main__":
    print("Testing LLM Router — all call types\n" + "=" * 40)

    test_msgs = [{"role": "user", "content": "Reply with just OK"}]

    for call_type in CHAINS.keys():
        result = route(call_type, test_msgs, max_tokens=10)
        status = "GOOD" if result["success"] else "FAILED"
        provider = result.get("provider_used", "none")
        print(f"{call_type:25s} → {status} via {provider}")

    print("\n--- Provider Health ---")
    health = health_check_all()
    for provider, status in health.items():
        print(f"  {provider:15s}: {status}")
