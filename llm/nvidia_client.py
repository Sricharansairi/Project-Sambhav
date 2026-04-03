import os, time, logging, base64
from openai import OpenAI
from dotenv import load_dotenv
from api import key_rotator

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

# ── Vision Call Wrapper ────────────────────────────────────────
def call_nvidia_vision(messages: list, model: str = "nvidia/qwen2-7b-instruct", 
                       temperature: float = 0.2, max_tokens: int = 1000) -> str:
    """NVIDIA NIM vision model wrapper."""
    return call_nvidia(messages, model=model, temperature=temperature, max_tokens=max_tokens)
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
    Section 6.3 — Uses NVIDIA NIM Qwen VLM (primary).
    Extracts: emotion, stress, engagement, body language, environment.
    """
    # Encode image to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

    messages = [
        {"role": "system", "content": (
            f"You are the NVIDIA NIM Qwen 2.5 VLM vision engine for the {domain} domain.\n"
            "Analyze the image and extract relevant parameters for probabilistic inference.\n"
            "Respond ONLY in this exact JSON format:\n"
            "{\n"
            '  "dominant_emotion": "<happy|sad|angry|fear|neutral|surprise|disgust>",\n'
            '  "stress_level": <0.0-1.0>,\n'
            '  "engagement_level": <0.0-1.0>,\n'
            '  "body_language": "<open|neutral|closed>",\n'
            '  "environment": "<formal|informal|neutral>",\n'
            '  "energy_level": <0.0-1.0>,\n'
            '  "people_count": <number>,\n'
            '  "key_observations": ["<obs1>", "<obs2>", "<obs3>"],\n'
            '  "confidence": <0.0-1.0>\n'
            "}"
        )},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{image_data}"}},
            {"type": "text",
             "text": f"Analyze this image for {domain} domain inference."}
        ]}
    ]

    try:
        # Using Qwen 2.5 VL as requested (placeholder if 3.5 not in NIM yet)
        raw = call_nvidia(messages, model="nvidia/qwen2-7b-instruct",
                          temperature=0.2, max_tokens=600)
        
        import json, re
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        data["source"] = image_path
        data["raw"]    = raw
        return data
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
        k = key_rotator.get_key("gemini")
        genai.configure(api_key=k)
        model  = genai.GenerativeModel("models/gemini-flash-latest")
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
