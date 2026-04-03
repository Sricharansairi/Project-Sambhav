import os, logging, tempfile
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Whisper transcription ─────────────────────────────────────
def _groq_whisper(audio_path: str) -> dict:
    """Use Groq API for high-speed Whisper transcription."""
    try:
        from groq import Groq
        from api.key_rotator import get_key
        key = get_key("groq")
        if not key: return {"success": False, "error": "No Groq key"}
        
        client = Groq(api_key=key)
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
        return {"text": transcription.text, "provider": "groq_whisper", "success": True}
    except Exception as e:
        logger.warning(f"Groq Whisper failed: {e}")
        return {"success": False, "error": str(e)}

def transcribe(audio_path: str) -> dict:
    """
    Transcribe audio — cascade:
    1. Groq Whisper Large V3 (FREE with existing keys)
    2. OpenAI Whisper (if key available)
    3. Local Whisper (if installed)
    """
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}
    
    # Try Groq Whisper first — FREE!
    result = _groq_whisper(audio_path)
    if result.get("success"):
        return result
    
    # Try OpenAI Whisper
    try:
        from openai import OpenAI
        key = os.getenv("WHISPER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if key:
            client = OpenAI(api_key=key)
            with open(audio_path, "rb") as f:
                r = client.audio.transcriptions.create(
                    model="whisper-1", file=f, language="en")
            return {"text": r.text, "provider": "openai_whisper", "success": True}
    except Exception as e:
        logger.warning(f"OpenAI Whisper failed: {e}")
    
    # Try AssemblyAI fallback
    try:
        import assemblyai as aai
        from api.key_rotator import get_key
        aai.settings.api_key = get_key("assemblyai")
        transcriber = aai.Transcriber()
        result = transcriber.transcribe(audio_path)
        if result.status == aai.TranscriptStatus.completed:
            return {"text": result.text, "provider": "assemblyai", "success": True}
    except Exception as e:
        logger.warning(f"AssemblyAI failed: {e}")

    # Final fallback — local
    return _local_whisper(audio_path)

def _local_whisper(audio_path: str) -> dict:
    """Fallback — local whisper if installed."""
    try:
        import whisper
        model  = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return {"text": result["text"], "provider": "local_whisper",
                "success": True}
    except Exception as e:
        logger.error(f"Local whisper also failed: {e}")
        return {"text": "", "provider": "none",
                "success": False, "error": str(e)}

# ── Extract video audio ───────────────────────────────────────
def extract_audio(video_path: str) -> str:
    """Extract audio track from video file."""
    try:
        import subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-q:a", "0", "-map", "a", tmp.name, "-y"
        ], capture_output=True, check=True)
        return tmp.name
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return ""

# ── Analyze transcribed text ──────────────────────────────────
def analyze_voice_with_llm(text: str, domain: str) -> dict:
    """
    Extract domain-specific parameters from voice transcript.
    Section 6.3 — Hybrid input extraction.
    """
    try:
        from llm.router import route
        from api.endpoints.predict import _load_registry
        import json, re

        reg = _load_registry()
        domain_params = reg.get(domain, {}).get("parameters", {})
        
        param_info = ""
        if domain_params:
            for k, p in (domain_params.items() if isinstance(domain_params, dict) else enumerate(domain_params)):
                key = p.get("key") or k
                label = p.get("label", key)
                param_info += f"- {key}: {label}\n"

        messages = [
            {"role": "system", "content": (
                f"You are the Sambhav Voice Analysis engine for the {domain} domain.\n"
                f"Analyze the spoken transcript and extract relevant parameters.\n"
                f"Parameters to look for:\n{param_info or 'Any domain-relevant signals'}\n\n"
                "Respond ONLY in JSON format:\n"
                "{\n"
                '  "parameters": {"key": "value", ...},\n'
                '  "confidence": <0.0-1.0>,\n'
                '  "summary": "<1 sentence summary>"\n'
                "}"
            )},
            {"role": "user", "content": text}
        ]

        raw = route("safety_screen", messages, max_tokens=300, temperature=0.1)
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        return data
    except Exception as e:
        logger.error(f"Voice LLM analysis failed: {e}")
        return {"parameters": {}, "confidence": 0.0}

def analyze_transcript(text: str, domain: str) -> dict:
    """Run full NLP + LLM analysis on transcribed text."""
    if not text:
        return {"error": "Empty transcript"}
    try:
        # Feature engineering on transcript text
        from core.feature_engineer import (
            extract_linguistic_features,
            extract_sentiment_features
        )
        linguistic  = extract_linguistic_features(text)
        sentiment   = extract_sentiment_features(text)

        # Step 1 — Fast parameter extraction (Mode 3 Hybrid)
        llm_result = analyze_voice_with_llm(text, domain)

        return {
            "transcript":         text,
            "linguistic_features":linguistic,
            "sentiment_features": sentiment,
            "llm_confidence":     llm_result.get("confidence", 0.5),
            "summary":            llm_result.get("summary", ""),
            "inferred_parameters": {
                **linguistic,
                **sentiment,
                **llm_result.get("parameters", {}),
                "transcript_length": len(text.split()),
            }
        }
    except Exception as e:
        logger.error(f"Transcript analysis failed: {e}")
        return {"transcript": text, "error": str(e)}

# ── MAIN ENTRY POINT ──────────────────────────────────────────
def analyze_voice(audio_path: str, domain: str = "general") -> dict:
    """
    Full voice pipeline:
    1. Transcribe audio → text
    2. NLP feature extraction
    3. LLM analysis
    4. Return parameters ready for predictor
    """
    logger.info(f"Analyzing voice: {audio_path}, domain={domain}")

    # Step 1 — Transcribe
    transcript_result = transcribe(audio_path)
    if not transcript_result.get("success"):
        return {"error": transcript_result.get("error", "Transcription failed"),
                "path": audio_path}

    text = transcript_result["text"]
    logger.info(f"Transcribed {len(text)} chars")

    # Step 2+3 — Analyze transcript
    analysis = analyze_transcript(text, domain)
    analysis["provider"]   = transcript_result.get("provider")
    analysis["audio_path"] = audio_path

    return analysis

def analyze_voice_from_video(video_path: str, domain: str = "general") -> dict:
    """Extract audio from video then analyze."""
    audio_path = extract_audio(video_path)
    if not audio_path:
        return {"error": "Could not extract audio from video"}
    result = analyze_voice(audio_path, domain)
    # Cleanup temp audio
    try: os.unlink(audio_path)
    except: pass
    return result

if __name__ == "__main__":
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else "test.mp3"
    domain = sys.argv[2] if len(sys.argv) > 2 else "general"
    result = analyze_voice(path, domain)
    print(f"\n🎤 VOICE ANALYSIS:")
    print(f"  Transcript : {result.get('transcript','')[:100]}...")
    print(f"  LLM Prob   : {result.get('llm_probability')}")
    print(f"  Confidence : {result.get('llm_confidence')}")
