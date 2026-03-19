import os, time, logging, requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/chat/completions"

MODELS = {
    "large": "qwen-3-235b-a22b-instruct-2507",
    "fast":  "llama3.1-8b",
}

def _load_keys():
    keys = []
    for i in range(1, 10):
        k = os.getenv(f"CEREBRAS_API_KEY_{i}")
        if k and k not in keys:
            keys.append(k)
    if not keys:
        raise ValueError("No Cerebras API keys found in .env")
    return keys

KEYS = _load_keys()
_key_index = 0

def _get_key():
    global _key_index
    key = KEYS[_key_index % len(KEYS)]
    _key_index += 1
    return key

def call_cerebras(
    messages: list,
    model: str = "large",
    temperature: float = 0.2,
    max_tokens: int = 1000,
    retries: int = 3
) -> str:
    model_name = MODELS.get(model, MODELS["large"])
    for attempt in range(retries):
        key = _get_key()
        try:
            r = requests.post(
                CEREBRAS_BASE_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=20
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            wait = 2 ** attempt
            logger.warning(f"Cerebras attempt {attempt+1} failed: {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"Cerebras error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Cerebras call failed after {retries} attempts")

def health_check() -> dict:
    try:
        resp = call_cerebras([{"role": "user", "content": "Reply OK"}], max_tokens=5)
        return {"status": "ok", "response": resp, "keys_loaded": len(KEYS)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("Testing Cerebras client...")
    print(health_check())
