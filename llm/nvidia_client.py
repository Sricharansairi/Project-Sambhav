import os, time, logging, base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Load NVIDIA NIM keys ──────────────────────────────────────
def _load_nvidia_keys():
    keys = []
    for i in range(1, 5):
        k = os.getenv(f"NVIDIA_API_KEY_{i}") or os.getenv("NVIDIA_API_KEY")
        if k and k not in keys:
            keys.append(k)
    return keys

NVIDIA_KEYS = _load_nvidia_keys()
_key_index  = 0
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"

def _get_client():
    global _key_index
    if not NVIDIA_KEYS:
        raise ValueError("No NVIDIA API keys found in .env")
    key = NVIDIA_KEYS[_key_index % len(NVIDIA_KEYS)]
    _key_index += 1
    return OpenAI(api_key=key, base_url=NVIDIA_BASE)

# ── Core text call ────────────────────────────────────────────
def call_nvidia(
    messages: list,
    model: str = "meta/llama-3.3-70b-instruct",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    retries: int = 3
) -> str:
    for attempt in range(retries):
        try:
            client  = _get_client()
            resp    = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"NVIDIA attempt {attempt+1} failed: {e}, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"NVIDIA NIM call failed after {retries} attempts")

# ── Vision — Image Analysis ───────────────────────────────────
def analyze_image(image_path: str, domain: str) -> dict:
    """
    Analyze an image and extract domain-relevant parameters.
    Uses Qwen VLM (primary) via NVIDIA NIM.
    """
    # Encode image to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

    messages = [
        {"role": "system", "content": (
            f"You are a vision analysis engine for the {domain} domain. "
            "Analyze the image and extract relevant parameters for probabilistic inference. "
            "Respond in this exact format:\n"
            "DOMINANT_EMOTION: <emotion>\n"
            "STRESS_LEVEL: <0.0-1.0>\n"
            "ENGAGEMENT_LEVEL: <0.0-1.0>\n"
            "BODY_LANGUAGE: <open|neutral|closed>\n"
            "ENVIRONMENT: <formal|informal|neutral>\n"
            "ENERGY_LEVEL: <0.0-1.0>\n"
            "PEOPLE_COUNT: <number>\n"
            "KEY_OBSERVATIONS: <comma separated top 3 observations>"
        )},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{image_data}"}},
            {"type": "text",
             "text": f"Analyze this image for {domain} domain inference."}
        ]}
    ]

    try:
        raw = call_nvidia(messages, model="meta/llama-3.2-90b-vision-instruct",
                          temperature=0.2, max_tokens=400)
        return _parse_vision_response(raw, image_path)
    except Exception as e:
        logger.warning(f"NVIDIA vision failed, using fallback: {e}")
        return _gemini_vision_fallback(image_path, domain)

def _parse_vision_response(raw: str, source: str) -> dict:
    result = {
        "source": source, "raw": raw,
        "dominant_emotion": "neutral",
        "stress_level": 0.5,
        "engagement_level": 0.5,
        "body_language": "neutral",
        "environment": "neutral",
        "energy_level": 0.5,
        "people_count": 1,
        "key_observations": []
    }
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("DOMINANT_EMOTION:"):
            result["dominant_emotion"] = line.split(":",1)[1].strip().lower()
        elif line.startswith("STRESS_LEVEL:"):
            try: result["stress_level"] = float(line.split(":")[1].strip())
            except: pass
        elif line.startswith("ENGAGEMENT_LEVEL:"):
            try: result["engagement_level"] = float(line.split(":")[1].strip())
            except: pass
        elif line.startswith("BODY_LANGUAGE:"):
            result["body_language"] = line.split(":",1)[1].strip().lower()
        elif line.startswith("ENVIRONMENT:"):
            result["environment"] = line.split(":",1)[1].strip().lower()
        elif line.startswith("ENERGY_LEVEL:"):
            try: result["energy_level"] = float(line.split(":")[1].strip())
            except: pass
        elif line.startswith("PEOPLE_COUNT:"):
            try: result["people_count"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("KEY_OBSERVATIONS:"):
            result["key_observations"] = [o.strip() for o in line.split(":",1)[1].split(",")]
    return result

# ── Document Analysis ─────────────────────────────────────────
def analyze_document(text: str, domain: str) -> dict:
    """
    Analyze a full document text and return domain-relevant parameters.
    Uses Llama 405B (1M context via NIM) — no chunking needed.
    """
    messages = [
        {"role": "system", "content": (
            f"You are a document analysis engine for {domain} inference. "
            "Extract key parameters from the document for probabilistic prediction. "
            "Respond in this exact format:\n"
            "DOMAIN_DETECTED: <domain>\n"
            "KEY_PARAMETERS: <param: value pairs, one per line>\n"
            "SUMMARY: <2 sentence document summary>\n"
            "CONFIDENCE: <HIGH|MODERATE|LOW>\n"
            "PREDICTION_QUESTION: <most relevant yes/no question to answer>"
        )},
        {"role": "user", "content": (
            f"Analyze this document for {domain} domain inference:\n\n"
            f"{text[:8000]}"  # Cap at 8k chars for safety
        )}
    ]
    try:
        raw = call_nvidia(messages, temperature=0.2, max_tokens=600)
        return _parse_document_response(raw)
    except Exception as e:
        logger.warning(f"NVIDIA document analysis failed: {e}")
        return {"error": str(e), "raw": ""}

def _parse_document_response(raw: str) -> dict:
    result = {
        "raw": raw,
        "domain_detected": "general",
        "key_parameters": {},
        "summary": "",
        "confidence": "MODERATE",
        "prediction_question": ""
    }
    lines = raw.split("\n")
    in_params = False
    for line in lines:
        line = line.strip()
        if line.startswith("DOMAIN_DETECTED:"):
            result["domain_detected"] = line.split(":",1)[1].strip()
            in_params = False
        elif line.startswith("KEY_PARAMETERS:"):
            in_params = True
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":",1)[1].strip()
            in_params = False
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":",1)[1].strip()
            in_params = False
        elif line.startswith("PREDICTION_QUESTION:"):
            result["prediction_question"] = line.split(":",1)[1].strip()
            in_params = False
        elif in_params and ":" in line:
            k, v = line.split(":", 1)
            result["key_parameters"][k.strip()] = v.strip()
    return result

# ── Gemini fallback for vision ────────────────────────────────
def _gemini_vision_fallback(image_path: str, domain: str) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY"))
        model  = genai.GenerativeModel("gemini-1.5-flash")
        import PIL.Image
        img    = PIL.Image.open(image_path)
        prompt = (f"Analyze this image for {domain} probabilistic inference. "
                  "Extract: emotion, stress level (0-1), engagement (0-1), "
                  "body language, environment formality, energy level (0-1).")
        resp   = model.generate_content([prompt, img])
        return {"source": image_path, "raw": resp.text,
                "fallback": "gemini", "stress_level": 0.5,
                "engagement_level": 0.5, "dominant_emotion": "neutral",
                "body_language": "neutral", "environment": "neutral",
                "energy_level": 0.5, "people_count": 1, "key_observations": []}
    except Exception as e:
        logger.error(f"Gemini fallback also failed: {e}")
        return {"source": image_path, "error": str(e),
                "stress_level": 0.5, "engagement_level": 0.5,
                "dominant_emotion": "neutral", "body_language": "neutral",
                "environment": "neutral", "energy_level": 0.5,
                "people_count": 1, "key_observations": []}

# ── Health check ──────────────────────────────────────────────
def health_check() -> dict:
    if not NVIDIA_KEYS:
        return {"status": "no_keys", "keys_loaded": 0}
    try:
        resp = call_nvidia(
            [{"role": "user", "content": "Reply with only: OK"}],
            max_tokens=5, temperature=0
        )
        return {"status": "ok", "response": resp, "keys_loaded": len(NVIDIA_KEYS)}
    except Exception as e:
        return {"status": "error", "error": str(e), "keys_loaded": len(NVIDIA_KEYS)}

if __name__ == "__main__":
    print("NVIDIA NIM Health Check:")
    print(health_check())
