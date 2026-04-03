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
    Analyze uploaded image and run auto-prediction (Mode 3 Hybrid).
    Extracts: emotion, stress, engagement, body language, environment.
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
        from core.predictor import predict, generate_outcomes
        
        # Step 1 — Vision analysis (P.03/P.04)
        result = analyze_image(tmp_path, domain)
        
        # Step 2 — Insufficient Info Check
        params = result.get("inferred_parameters", {})
        if result.get("confidence") == "LOW" or not params:
            return {
                "success": True,
                "insufficient_info": True,
                "reason": "The image is too blurry or lacks clear domain-relevant signals (e.g., face, body language).",
                "result": result
            }

        # Step 3 — Auto Prediction
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pred_future = executor.submit(predict, domain=domain, params=params)
            out_future  = executor.submit(generate_outcomes, domain=domain, parameters=params)
            
            prediction = pred_future.result()
            outcomes   = out_future.result()

        return {
            "success":    True,
            "result":     result,
            "prediction": prediction.to_dict(),
            "outcomes":   outcomes.get("outcomes", []),
            "insufficient_info": False,
            "disclaimer": "Sambhav vision is probabilistic. Always verify independently.",
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
    Analyze uploaded video and run auto-prediction.
    Extracts frame-by-frame emotion timeline + key moments.
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
        from core.predictor import predict, generate_outcomes
        
        # Step 1 — Video analysis (P.07/P.08/P.09)
        result = analyze_video(tmp_path, domain, mode)
        
        # Step 2 — Insufficient Info Check
        params = result.get("inferred_parameters", {})
        if not params or result.get("aggregated", {}).get("frame_reliability", 0) < 0.3:
            return {
                "success": True,
                "insufficient_info": True,
                "reason": "The video duration is too short or frames are unreadable.",
                "result": result
            }

        # Step 3 — Auto Prediction
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pred_future = executor.submit(predict, domain=domain, params=params)
            out_future  = executor.submit(generate_outcomes, domain=domain, parameters=params)
            
            prediction = pred_future.result()
            outcomes   = out_future.result()

        return {
            "success":    True,
            "result":     result,
            "prediction": prediction.to_dict(),
            "outcomes":   outcomes.get("outcomes", []),
            "insufficient_info": False,
            "disclaimer": "Video analysis is based on temporal emotional cues.",
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
    Analyze uploaded document (PDF/DOCX/TXT) and run prediction.
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
        from core.predictor import predict, generate_outcomes
        
        # Step 1 — Document analysis (P.15-P.19)
        result = analyze_document(tmp_path, domain)
        params = result.get("parameters", {})
        if not params or result.get("confidence") == "LOW":
            return {
                "success": True,
                "insufficient_info": True,
                "reason": "The document does not contain enough data points for a reliable {domain} prediction.",
                "result": result
            }

        # Step 3 — Auto Prediction
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pred_future = executor.submit(predict, domain=domain, params=params)
            out_future  = executor.submit(generate_outcomes, domain=domain, parameters=params)
            
            prediction = pred_future.result()
            outcomes   = out_future.result()

        return {
            "success":    True,
            "result":     result,
            "prediction": prediction.to_dict(),
            "outcomes":   outcomes.get("outcomes", []),
            "insufficient_info": False
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
    Transcribe and analyze voice input with auto-prediction.
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
        from core.predictor import predict, generate_outcomes
        
        # Step 1 — Voice analysis (P.11-P.14)
        result = analyze_voice(tmp_path, domain)
        
        # Step 2 — Insufficient Info Check
        params = result.get("inferred_parameters", {})
        if not params or not result.get("transcript"):
            return {
                "success": True,
                "insufficient_info": True,
                "reason": "Audio is silent or transcription failed.",
                "result": result
            }

        # Step 3 — Auto Prediction
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pred_future = executor.submit(predict, domain=domain, params=params)
            out_future  = executor.submit(generate_outcomes, domain=domain, parameters=params)
            
            prediction = pred_future.result()
            outcomes   = out_future.result()

        return {
            "success":    True,
            "result":     result,
            "prediction": prediction.to_dict(),
            "outcomes":   outcomes.get("outcomes", []),
            "insufficient_info": False
        }
    except Exception as e:
        logger.error(f"Voice analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass
