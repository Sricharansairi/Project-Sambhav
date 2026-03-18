import os, logging, shutil, tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from core.safety import check_hard_blocks

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg","image/png","image/webp","image/jpg"}
ALLOWED_VIDEO_TYPES = {"video/mp4","video/mpeg","video/quicktime","video/webm"}
ALLOWED_DOC_TYPES   = {"application/pdf","text/plain",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
MAX_FILE_MB = 50

def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to temp location."""
    suffix = "." + file.filename.split(".")[-1]
    tmp    = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    return tmp.name

def _check_size(file: UploadFile):
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024*1024)
    file.file.seek(0)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB (max {MAX_FILE_MB}MB)")

# ── POST /vision/image ────────────────────────────────────────
@router.post("/image")
async def analyze_image_endpoint(
    file:   UploadFile = File(...),
    domain: str        = Form(default="general"),
):
    """
    Analyze uploaded image.
    Extracts: emotion, stress, engagement, body language, environment.
    Uses Gemini Vision → DeepFace → MediaPipe.
    """
    logger.info(f"POST /vision/image domain={domain} file={file.filename}")

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {ALLOWED_IMAGE_TYPES}")

    _check_size(file)
    tmp_path = _save_upload(file)

    try:
        from vision.image_pipeline import analyze_image
        result = analyze_image(tmp_path, domain)
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ── POST /vision/video ────────────────────────────────────────
@router.post("/video")
async def analyze_video_endpoint(
    file:   UploadFile = File(...),
    domain: str        = Form(default="general"),
    mode:   str        = Form(default="standard"),
):
    """
    Analyze uploaded video.
    Extracts frame-by-frame emotion timeline + key moments.
    Modes: fast (15-20s) | standard (25-40s) | deep (60-90s)
    """
    logger.info(f"POST /vision/video domain={domain} mode={mode} file={file.filename}")

    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}")

    _check_size(file)
    tmp_path = _save_upload(file)

    try:
        from vision.video_pipeline import analyze_video
        result = analyze_video(tmp_path, domain, mode)
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ── POST /vision/document ─────────────────────────────────────
@router.post("/document")
async def analyze_document_endpoint(
    file:   UploadFile = File(...),
    domain: str        = Form(default="general"),
):
    """
    Analyze uploaded document (PDF/DOCX/TXT).
    Extracts text → LLM parameter extraction → prediction-ready output.
    """
    logger.info(f"POST /vision/document domain={domain} file={file.filename}")

    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}")

    _check_size(file)
    tmp_path = _save_upload(file)

    try:
        from vision.document_pipeline import analyze_document
        result = analyze_document(tmp_path, domain)

        # Safety check on extracted text
        if result.get("text_preview"):
            safety = check_hard_blocks(result["text_preview"])
            if not safety["safe"]:
                raise HTTPException(
                    status_code=400,
                    detail={"blocked": True,
                            "message": safety["message"]})

        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Document analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ── POST /vision/voice ────────────────────────────────────────
@router.post("/voice")
async def analyze_voice_endpoint(
    file:   UploadFile = File(...),
    domain: str        = Form(default="general"),
):
    """
    Transcribe and analyze voice input.
    Uses Whisper → NLP features → LLM analysis.
    """
    logger.info(f"POST /vision/voice domain={domain} file={file.filename}")

    allowed_audio = {"audio/mpeg","audio/mp3","audio/wav",
                     "audio/ogg","audio/m4a","audio/webm"}
    if file.content_type not in allowed_audio:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio type: {file.content_type}")

    _check_size(file)
    tmp_path = _save_upload(file)

    try:
        from vision.voice_pipeline import analyze_voice
        result = analyze_voice(tmp_path, domain)
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Voice analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass
