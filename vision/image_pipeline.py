import os, base64, logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Key loaders ───────────────────────────────────────────────
def _gemini_keys():
    keys = []
    for i in range(1, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if k and k not in keys: keys.append(k)
    return keys

GEMINI_KEYS = _gemini_keys()
_gem_idx    = 0

def _get_gemini():
    global _gem_idx
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEYS[_gem_idx % len(GEMINI_KEYS)])
    _gem_idx += 1
    return genai.GenerativeModel("gemini-2.0-flash")

# ── Base64 encoder ────────────────────────────────────────────
def _encode_image(path: str) -> tuple:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext  = path.split(".")[-1].lower()
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
            "png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")
    return data, mime

# ── Parse vision response ─────────────────────────────────────
def _parse(raw: str, source: str) -> dict:
    result = {
        "source":           source,
        "raw":              raw,
        "dominant_emotion": "neutral",
        "stress_level":     0.5,
        "engagement_level": 0.5,
        "body_language":    "neutral",
        "environment":      "neutral",
        "energy_level":     0.5,
        "people_count":     1,
        "key_observations": [],
        "confidence":       "MODERATE",
    }
    for line in raw.split("\n"):
        line = line.strip()
        if   line.startswith("DOMINANT_EMOTION:"):
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
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":",1)[1].strip()
    return result

VISION_PROMPT = """Analyze this image for probabilistic inference.
Respond ONLY in this exact format:
DOMINANT_EMOTION: <happy|sad|angry|fear|neutral|surprise|disgust>
STRESS_LEVEL: <0.0-1.0>
ENGAGEMENT_LEVEL: <0.0-1.0>
BODY_LANGUAGE: <open|neutral|closed>
ENVIRONMENT: <formal|informal|neutral>
ENERGY_LEVEL: <0.0-1.0>
PEOPLE_COUNT: <number>
KEY_OBSERVATIONS: <observation1, observation2, observation3>
CONFIDENCE: <HIGH|MODERATE|LOW>"""

# ── Primary — Gemini Vision ───────────────────────────────────
def _analyze_gemini(image_path: str, domain: str) -> dict:
    import PIL.Image
    img   = PIL.Image.open(image_path)
    model = _get_gemini()
    prompt = f"Domain context: {domain}\n\n{VISION_PROMPT}"
    resp  = model.generate_content([prompt, img])
    return _parse(resp.text, image_path)

# ── Fallback — NVIDIA NIM ─────────────────────────────────────
def _analyze_nvidia(image_path: str, domain: str) -> dict:
    from llm.nvidia_client import analyze_image
    return analyze_image(image_path, domain)

# ── DeepFace emotion (local, no API needed) ───────────────────
def _deepface_emotion(image_path: str) -> dict:
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(image_path, actions=['emotion'],
                                  enforce_detection=False)
        if isinstance(result, list): result = result[0]
        dominant = result.get("dominant_emotion", "neutral")
        emotions = result.get("emotion", {})
        return {"dominant_emotion": dominant, "emotions": emotions,
                "deepface_ok": True}
    except Exception as e:
        logger.warning(f"DeepFace failed: {e}")
        return {"dominant_emotion": "neutral", "deepface_ok": False}

# ── MediaPipe pose (local, no API needed) ─────────────────────
def _mediapipe_pose(image_path: str) -> dict:
    try:
        import mediapipe as mp
        import cv2
        mp_pose    = mp.solutions.pose
        img        = cv2.imread(image_path)
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Openness — distance between wrists
            left_wrist  = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            openness = min(1.0, abs(left_wrist.x - right_wrist.x) * 2)
            return {"body_openness": round(openness, 3),
                    "pose_detected": True}
        return {"body_openness": 0.5, "pose_detected": False}
    except Exception as e:
        logger.warning(f"MediaPipe failed: {e}")
        return {"body_openness": 0.5, "pose_detected": False}

# ── MAIN ENTRY POINT ──────────────────────────────────────────
def analyze_image(image_path: str, domain: str = "general") -> dict:
    """
    Full image analysis pipeline.
    1. Gemini Vision (primary) — extracts all parameters
    2. DeepFace — facial emotion (local, free)
    3. MediaPipe — body language (local, free)
    4. NVIDIA NIM fallback if Gemini fails
    Returns merged result dict ready for predictor.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    logger.info(f"Analyzing image: {image_path} for domain: {domain}")
    result = {}

    # Step 1 — Gemini Vision
    try:
        result = _analyze_gemini(image_path, domain)
        result["vision_provider"] = "gemini"
        logger.info("Gemini vision: OK")
    except Exception as e:
        logger.warning(f"Gemini vision failed: {e}, trying NVIDIA...")
        try:
            result = _analyze_nvidia(image_path, domain)
            result["vision_provider"] = "nvidia"
        except Exception as e2:
            logger.error(f"All vision providers failed: {e2}")
            result = _parse("", image_path)
            result["vision_provider"] = "fallback_defaults"

    # Step 2 — DeepFace (override emotion if detected)
    df_result = _deepface_emotion(image_path)
    if df_result.get("deepface_ok"):
        result["dominant_emotion"] = df_result["dominant_emotion"]
        result["emotion_breakdown"] = df_result.get("emotions", {})
        result["deepface_used"]     = True

    # Step 3 — MediaPipe (add body openness)
    mp_result = _mediapipe_pose(image_path)
    result["body_openness"] = mp_result.get("body_openness", 0.5)
    result["pose_detected"] = mp_result.get("pose_detected", False)

    # Step 4 — Derive inference parameters from vision
    result["inferred_parameters"] = {
        "stress_level":     result.get("stress_level", 0.5),
        "engagement_level": result.get("engagement_level", 0.5),
        "energy_level":     result.get("energy_level", 0.5),
        "body_openness":    result.get("body_openness", 0.5),
        "dominant_emotion": result.get("dominant_emotion", "neutral"),
        "environment":      result.get("environment", "neutral"),
        "people_count":     result.get("people_count", 1),
    }

    logger.info(f"Image analysis complete: {result.get('dominant_emotion')} "
                f"stress={result.get('stress_level')} "
                f"engage={result.get('engagement_level')}")
    return result


if __name__ == "__main__":
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    domain = sys.argv[2] if len(sys.argv) > 2 else "general"
    result = analyze_image(path, domain)
    print("\n📸 IMAGE ANALYSIS RESULT:")
    for k, v in result.items():
        if k not in ["raw", "source"]:
            print(f"  {k}: {v}")
