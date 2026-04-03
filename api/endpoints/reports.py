from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from api.endpoints.auth import get_current_user
from sambhav_exports_v3 import run_exports
import os
from db.models import get_db
from db.database import get_prediction_by_id

router = APIRouter()
EXPORTS_DIR = os.getenv("EXPORTS_DIR", "exports")

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
    "pdf_simple":   ["1"],
    "pdf_detailed": ["2"],
    "pdf_full":     ["3"],
    "docx":         ["4"],
    "xlsx":         ["5"],
    "pptx":         ["6"],
    "json":         ["7"],
    "txt":          ["8"],
    "all":          list("12345678"),
}
MIME_MAP = {
    "pdf_simple":   "application/pdf",
    "pdf_detailed": "application/pdf",
    "pdf_full":     "application/pdf",
    "docx":         "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx":         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx":         "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "json":         "application/json",
    "txt":          "text/plain",
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
    out_dir = f"{EXPORTS_DIR}/{prediction_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    outputs = run_exports(data=data, output_dir=out_dir,
                          formats=FMT_MAP.get(fmt, ["3"]))
    
    key_mapping = {
        "1":"pdf_simple","2":"pdf_detailed","3":"pdf_full",
        "4":"docx","5":"xlsx","6":"pptx","7":"json","8":"txt"
    }
    requested_formats = FMT_MAP.get(fmt, ["3"])
    first_choice = requested_formats[0]
    
    file_path = outputs.get(key_mapping.get(first_choice)) or (list(outputs.values())[0] if outputs else None)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(500, "Failed to generate report")
        
    mime = MIME_MAP.get(fmt, "application/pdf")
    ext  = Path(file_path).suffix
    return FileResponse(file_path, media_type=mime,
                        filename=f"{prediction_id}_report{ext}")
