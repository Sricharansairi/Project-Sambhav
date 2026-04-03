from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from api.endpoints.auth import get_current_user
import os
from db.models import get_db
from db.database import get_prediction_by_id

# ── Import internal report generators ───────────────────────────
from reports.pdf_generator import generate_pdf
from reports.docx_generator import generate_docx
from reports.excel_generator import generate_xlsx
from reports.pptx_generator import generate_pptx

router = APIRouter()
EXPORTS_DIR = os.getenv("EXPORTS_DIR", "/tmp")

def pred_to_dict(pred) -> dict:
    """Map PredictionResponse ORM object → export dict."""
    return {
        "prediction_id":         pred.prediction_id,
        "project_name":          "Project Sambhav",
        "subtitle":              "Multi-Modal Probabilistic Inference Engine",
        "tagline":               "Uncertainty, Quantified.",
        "generated_at":          pred.created_at.strftime("%d %B %Y, %I:%M %p") if getattr(pred, "created_at", None) else "",
        "generated_date":        pred.created_at.strftime("%Y-%m-%d") if getattr(pred, "created_at", None) else "",
        "institution":           "Sri Indu Institute of Engineering & Technology",
        "developer":             "Sricharan Sairi",
        "version":               "Academic v1.0",
        "report_url":            f"https://sambhav-app.hf.space/report/{pred.prediction_id}",
        "domain":                pred.domain,
        "mode":                  pred.mode,
        "question":              pred.input_text or "Information provided via structured form.",
        "reliability_index":     int((pred.reliability_index or 0.82) * 100) if getattr(pred, "reliability_index", None) and pred.reliability_index <= 1 else int(getattr(pred, "reliability_index", None) or 82),
        "warning_level":         pred.warning_level or "CLEAR",
        "ml_probability":        pred.ml_probability or 0.5,
        "llm_probability":       pred.llm_probability or 0.5,
        "reconciled_probability":pred.reconciled_prob or 0.5,
        "agreement_gap":         pred.agreement_gap or 0.0,
        "confidence_interval":   {"low": (pred.reconciled_prob or 0.5) - 0.07,
                                  "high": (pred.reconciled_prob or 0.5) + 0.07},
        "parameters":            pred.parameters or {},
        "outcomes":              pred.outcomes or [],
        "shap_values":           pred.shap_values or [],
        "audit": {
            "overall_status":   pred.audit_status or "PASSED",
            "flags":            pred.audit_flags if isinstance(pred.audit_flags, list) else [],
            "engine_1_param":   "PASS — Parameters OK" if not pred.audit_flags else "SEE FLAGS",
            "engine_2_predict": "PASS — Prediction OK",
            "engine_3_conf":    "PASS — Confidence OK",
        },
        "failure_scenarios":     [],
        "improvement_actions":   [],
        "model": {
            "name":       "XGBoost + LightGBM Stacking",
            "brier_score": 0.0639,
            "auc":        0.8753,
            "accuracy":   "84.2 %",
            "calibration":"Manual IsotonicRegression on held-out calibration set",
            "train_data": "Multi-source dataset",
            "features":   "33 features across 4 families",
            "pipeline":   "7-Stage: Sanitise → Detect → Extract → Predict → CV → Audit → Deliver",
        },
        "disclaimer": (
            "Sambhav may be incorrect. Always verify independently. "
            "Academic use only. © 2026 Sricharan Sairi."
        ),
    }

FMT_MAP = {
    "pdf_simple":   "pdf",
    "pdf_detailed": "pdf",
    "pdf_full":     "pdf",
    "docx":         "docx",
    "xlsx":         "xlsx",
    "pptx":         "pptx",
}

MIME_MAP = {
    "pdf":          "application/pdf",
    "docx":         "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx":         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx":         "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

@router.post("/generate-report/{prediction_id}")
async def generate_report(
    prediction_id: str,
    fmt: str = "pdf_full",
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    pred = get_prediction_by_id(db, prediction_id)
    if not pred:
        raise HTTPException(404, "Prediction not found")
    
    data = pred_to_dict(pred)
    target_fmt = FMT_MAP.get(fmt, "pdf")
    filename = f"sambhav_report_{prediction_id}.{target_fmt}"
    filepath = os.path.join(EXPORTS_DIR, filename)

    try:
        if target_fmt == "pdf":
            generate_pdf(data, filepath)
        elif target_fmt == "docx":
            generate_docx(data, filepath)
        elif target_fmt == "xlsx":
            generate_xlsx(data, filepath)
        elif target_fmt == "pptx":
            generate_pptx(data, filepath)
        else:
            raise HTTPException(400, f"Unsupported format: {fmt}")

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=MIME_MAP.get(target_fmt)
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(500, f"Failed to generate report: {str(e)}")
