import os, logging, tempfile
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Whisper transcription ─────────────────────────────────────
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

        # LLM parameter extraction
        from llm.groq_client import llm_predict
        llm_result  = llm_predict(domain, {"transcript": text[:500]},
                       f"Based on this spoken text, what is the probability of a positive outcome?")

        return {
            "transcript":         text,
            "linguistic_features":linguistic,
            "sentiment_features": sentiment,
            "llm_probability":    llm_result.get("probability"),
            "llm_confidence":     llm_result.get("confidence"),
            "llm_reasoning":      llm_result.get("reasoning"),
            "inferred_parameters": {
                **linguistic,
                **sentiment,
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
