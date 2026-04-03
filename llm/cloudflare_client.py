import os, time, logging, requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MODELS = {
    "large":     "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "reasoning": "@cf/deepseek-ai/deepseek-r1-distill-llama-70b",
    "fast":      "@cf/meta/llama-3.2-3b-instruct",
}

def _load_keys():
    pairs = []
    for i in range(1, 10):
        token = os.getenv(f"CLOUDFLARE_API_KEY_{i}")
        account_id = os.getenv(f"CLOUDFLARE_ACCOUNT_ID_{i}")
        if token and account_id:
            pairs.append((token, account_id))
    if not pairs:
        raise ValueError("No Cloudflare keys found in .env")
    return pairs

PAIRS = _load_keys()
_index = 0

def _get_pair():
    global _index
    pair = PAIRS[_index % len(PAIRS)]
    _index += 1
    return pair

def call_cloudflare(
    messages: list,
    model: str = "large",
    temperature: float = 0.2,
    max_tokens: int = 1000,
    retries: int = 3
) -> str:
    model_name = MODELS.get(model, MODELS["large"])
    for attempt in range(retries):
        token, account_id = _get_pair()
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": max_tokens},
                timeout=20
            )
            if r.status_code == 200:
                return r.json()["result"]["response"].strip()
            wait = 2 ** attempt
            logger.warning(f"Cloudflare attempt {attempt+1} failed: {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"Cloudflare error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Cloudflare call failed after {retries} attempts")

def health_check() -> dict:
    try:
        resp = call_cloudflare([{"role": "user", "content": "Reply OK"}], max_tokens=5)
        return {"status": "ok", "response": resp, "pairs_loaded": len(PAIRS)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("Testing Cloudflare client...")
    print(health_check())
