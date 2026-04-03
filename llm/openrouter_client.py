import os, time, logging
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

def _load_keys():
    keys = []
    for i in range(1, 15):
        k = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if k and k not in keys: keys.append(k)
    return keys

OPENROUTER_KEYS = _load_keys()
_key_index      = 0

def _get_client():
    global _key_index
    if not OPENROUTER_KEYS:
        raise ValueError("No OpenRouter keys found")
    key = OPENROUTER_KEYS[_key_index % len(OPENROUTER_KEYS)]
    _key_index += 1
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

def call_openrouter(
    messages: list,
    model:    str   = "meta-llama/llama-3.3-70b-instruct:free",
    temperature: float = 0.3,
    max_tokens:  int   = 1000,
    retries:     int   = 3,
) -> str:
    for attempt in range(retries):
        try:
            client = _get_client()
            resp   = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                timeout=15)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"OpenRouter attempt {attempt+1}: {e}, retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("OpenRouter failed after retries")

def health_check() -> dict:
    try:
        resp = call_openrouter(
            [{"role":"user","content":"Reply with only: OK"}],
            max_tokens=5, temperature=0)
        return {"status":"ok","response":resp,"keys":len(OPENROUTER_KEYS)}
    except Exception as e:
        return {"status":"error","error":str(e),"keys":len(OPENROUTER_KEYS)}

if __name__ == "__main__":
    print("OpenRouter health check:")
    print(health_check())
