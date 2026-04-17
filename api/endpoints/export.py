"""
api/endpoints/export.py — Full export engine for Project Sambhav.
Supports: PDF, Word, Excel, JSON, CSV, XML, PNG (chart), API Link.
"""
import os, json, io, logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session
from db.models import get_db
from api.endpoints.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

EXPORTS_DIR = os.getenv("EXPORTS_DIR", "/tmp/sambhav_exports")
try:
    os.makedirs(EXPORTS_DIR, exist_ok=True)
except PermissionError:
    # Fallback for restricted environments like HuggingFace Spaces
    EXPORTS_DIR = "/tmp"
    logger.warning(f"Permission denied for exports directory. Falling back to {EXPORTS_DIR}")

BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


def _fmt_pct(v, default: str = "—") -> str:
    """Safely format a 0-1 float as a percentage string. Returns default if value is None/invalid."""
    try:
        if v is None:
            return default
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return default


def _fmt_pct0(v, default: str = "—") -> str:
    """Safely format a 0-1 float as a rounded percentage (no decimal). Returns default if None/invalid."""
    try:
        if v is None:
            return default
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return default


def _fmt_float(v, fmt: str = "+.4f", default: str = "0.0000") -> str:
    """Safely format a float. Returns default if value is None/invalid."""
    try:
        if v is None:
            return default
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return default


def _get_outcomes_list(data: dict) -> list:
    """Normalize outcomes to a list of (label, probability_pct) tuples regardless of source format."""
    # Prefer outcomes_list (set by frontend with full outcome objects)
    outcomes_list = data.get("outcomes_list", [])
    if outcomes_list and isinstance(outcomes_list, list):
        result = []
        for o in outcomes_list:
            if isinstance(o, dict):
                label = o.get("outcome", str(o))
                prob_raw = o.get("probability", 0)
                # probability may be 0-100 (frontend) or 0-1 (backend dict)
                prob_pct = f"{float(prob_raw):.1f}%" if float(prob_raw) > 1 else f"{float(prob_raw)*100:.1f}%"
                reasoning = o.get("reasoning", "")
                result.append((label, prob_pct, reasoning))
        if result:
            return result

    # Fallback: outcomes dict from prediction result
    outcomes = data.get("outcomes", {})
    if isinstance(outcomes, dict):
        return [(label, _fmt_pct(prob), "") for label, prob in outcomes.items()]
    return []


class ExportRequest(BaseModel):
    prediction_id: Optional[str] = Field(None, example="SMB-2026-00001")
    domain:        Optional[str] = None
    parameters:    Optional[dict]= {}
    result:        Optional[dict]= {}   # Full prediction result dict
    question:      Optional[str] = None
    format:        str           = Field("json", example="pdf")


def _get_prediction_data(req: ExportRequest, db: Session) -> dict:
    """Retrieve prediction data — from DB if prediction_id, else from req.result."""
    if req.prediction_id:
        try:
            from db.database import get_prediction
            pred = get_prediction(db, req.prediction_id)
            if pred:
                return pred
        except Exception as e:
            logger.warning(f"DB fetch failed: {e}")
    # Fall back to inline result
    return {
        "prediction_id": req.prediction_id or "SMB-EXPORT",
        "domain":        req.domain or "unknown",
        "question":      req.question or "",
        "parameters":    req.parameters or {},
        **(req.result or {}),
    }


# ── POST /export/json ─────────────────────────────────────────
@router.post("/json")
async def export_json(req: ExportRequest, db: Session = Depends(get_db)):
    data = _get_prediction_data(req, db)
    content = json.dumps(data, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.json"}
    )


# ── POST /export/csv ──────────────────────────────────────────
@router.post("/csv")
async def export_csv(req: ExportRequest, db: Session = Depends(get_db)):
    import csv
    data = _get_prediction_data(req, db)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Field", "Value"])
    writer.writerow(["Prediction ID",   data.get("prediction_id", "")])
    writer.writerow(["Domain",          data.get("domain", "")])
    writer.writerow(["Question",        data.get("question", "")])
    writer.writerow(["Final Probability", _fmt_pct(data.get("final_probability"))])
    writer.writerow(["ML Probability Layer",  _fmt_pct(data.get("ml_probability"))])
    writer.writerow(["LLM Inference Layer", _fmt_pct(data.get("llm_probability"))])
    writer.writerow(["Reliability Index", _fmt_pct(data.get("reliability_index"))])
    writer.writerow(["Confidence Tier", data.get("confidence_tier", "")])
    writer.writerow(["Agreement Gap",   _fmt_pct(data.get("gap"))])
    writer.writerow(["Warning Level",   data.get("warning_level", "CLEAR")])
    writer.writerow(["Mode",            data.get("mode", "")])

    writer.writerow([])
    writer.writerow(["Multi-Outcome Predictions", "Probability", "Reasoning"])
    for label, prob_pct, reasoning in _get_outcomes_list(data):
        writer.writerow([label, prob_pct, reasoning])

    writer.writerow([])
    writer.writerow(["Input Parameter", "Value"])
    for k, v in (data.get("raw_parameters") or data.get("parameters") or {}).items():
        writer.writerow([k, v])

    shap = data.get("shap_values", {})
    if shap:
        writer.writerow([])
        writer.writerow(["SHAP Feature Contribution", "Impact Score"])
        for k, v in shap.items():
            writer.writerow([k, _fmt_float(v)])

    content = output.getvalue().encode()
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.csv"}
    )


# ── POST /export/xml ──────────────────────────────────────────
@router.post("/xml")
async def export_xml(req: ExportRequest, db: Session = Depends(get_db)):
    data = _get_prediction_data(req, db)

    def _val(v):
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<SambhavPrediction>"]
    lines.append(f"  <PredictionId>{_val(data.get('prediction_id',''))}</PredictionId>")
    lines.append(f"  <Domain>{_val(data.get('domain',''))}</Domain>")
    lines.append(f"  <Question>{_val(data.get('question',''))}</Question>")
    lines.append(f"  <FinalProbability>{_val(_fmt_pct(data.get('final_probability')))}</FinalProbability>")
    lines.append(f"  <MLProbabilityLayer>{_val(_fmt_pct(data.get('ml_probability')))}</MLProbabilityLayer>")
    lines.append(f"  <LLMInferenceLayer>{_val(_fmt_pct(data.get('llm_probability')))}</LLMInferenceLayer>")
    lines.append(f"  <ReliabilityIndex>{_val(_fmt_pct(data.get('reliability_index')))}</ReliabilityIndex>")
    lines.append(f"  <ConfidenceTier>{_val(data.get('confidence_tier',''))}</ConfidenceTier>")
    lines.append(f"  <AgreementGap>{_val(_fmt_pct(data.get('gap')))}</AgreementGap>")
    lines.append(f"  <WarningLevel>{_val(data.get('warning_level','CLEAR'))}</WarningLevel>")
    lines.append(f"  <Mode>{_val(data.get('mode',''))}</Mode>")
    
    lines.append("  <Outcomes>")
    outcomes = data.get("outcomes", {})
    if isinstance(outcomes, dict):
        for label, prob in outcomes.items():
            lines.append(f"    <Outcome label=\"{_val(label)}\">{_val(_fmt_pct(prob))}</Outcome>")
    lines.append("  </Outcomes>")

    lines.append("  <Parameters>")
    for k, v in (data.get("raw_parameters") or data.get("parameters") or {}).items():
        lines.append(f"    <{k}>{_val(v)}</{k}>")
    lines.append("  </Parameters>")
    lines.append("  <SHAPValues>")
    for k, v in (data.get("shap_values") or {}).items():
        lines.append(f"    <Feature name=\"{_val(k)}\" impact=\"{_val(_fmt_float(v))}\" />")
    lines.append("  </SHAPValues>")
    lines.append("</SambhavPrediction>")

    content = "\n".join(lines).encode()
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/xml",
        headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.xml"}
    )


# ── POST /export/pdf ──────────────────────────────────────────
@router.post("/pdf")
async def export_pdf(req: ExportRequest, db: Session = Depends(get_db)):
    try:
        from reportlab.lib.pagesizes import A4
        import datetime
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import cm

        data = _get_prediction_data(req, db)
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)

        styles = getSampleStyleSheet()
        logo_style    = ParagraphStyle("logo", parent=styles["Heading1"],
                                       textColor=colors.HexColor("#ffffff"), fontSize=22, spaceAfter=2,
                                       fontName="Helvetica-Bold")
        tagline_style = ParagraphStyle("tagline", parent=styles["Normal"],
                                       textColor=colors.HexColor("#888888"), fontSize=9, spaceAfter=4)
        h2_style      = ParagraphStyle("h2", parent=styles["Heading2"],
                                       textColor=colors.HexColor("#cccccc"), fontSize=13, spaceBefore=14, spaceAfter=6,
                                       fontName="Helvetica-Bold")
        label_style   = ParagraphStyle("label", parent=styles["Normal"],
                                       textColor=colors.HexColor("#999999"), fontSize=9)

        TABLE_STYLE = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.HexColor("#a0aec0")),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1,-1), 9),
            ("TEXTCOLOR",     (0, 1), (-1,-1), colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS",(0, 1), (-1,-1), [colors.HexColor("#0d0d1a"), colors.HexColor("#111827")]),
            ("GRID",          (0, 0), (-1,-1), 0.3, colors.HexColor("#2d3748")),
            ("PADDING",       (0, 0), (-1,-1), 6),
            ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
        ])

        story = []

        # ── Header Banner ──────────────────────────────────────────
        header_bg = Table([[Paragraph("⚡ Project Sambhav", logo_style)]],
                           colWidths=[17*cm])
        header_bg.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#0a0a1a")),
            ("PADDING",     (0,0), (-1,-1), 12),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(header_bg)
        story.append(Paragraph("Uncertainty, Quantified — Multi-Modal Probabilistic Inference Engine", tagline_style))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ·  "
            f"ID: {data.get('prediction_id', 'N/A')}",
            label_style
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2d3748")))
        story.append(Spacer(1, 0.3*cm))

        # ── Prediction Summary ───────────────────────────────────────
        story.append(Paragraph("Prediction Summary", h2_style))
        fp = data.get("final_probability", 0)
        ri = data.get("reliability_index", 0)
        meta_rows = [
            ["Field", "Value"],
            ["Domain",           str(data.get("domain", "")).upper()],
            ["Question",         str(data.get("question", ""))],
            ["Mode",             str(data.get("mode", ""))],
            ["Final Probability",_fmt_pct(fp)],
            ["ML Layer",         _fmt_pct(data.get('ml_probability'))],
            ["LLM Layer",        _fmt_pct(data.get('llm_probability'))],
            ["Reliability Index",_fmt_pct0(ri)],
            ["Confidence Tier",  str(data.get("confidence_tier", ""))],
            ["ML-LLM Gap",       _fmt_pct(data.get('gap'))],
        ]
        meta_t = Table(meta_rows, colWidths=[5*cm, 12*cm])
        meta_t.setStyle(TABLE_STYLE)
        story.append(meta_t)
        story.append(Spacer(1, 0.4*cm))

        # ── Multi-Outcome Predictions ────────────────────────────────
        outcome_items = _get_outcomes_list(data)
        if outcome_items:
            story.append(Paragraph("Multi-Outcome Predictions", h2_style))
            o_rows = [["Outcome", "Probability", "Reasoning"]] + [
                [label, prob_pct, (reasoning[:80] + "...") if len(reasoning) > 80 else reasoning]
                for label, prob_pct, reasoning in outcome_items
            ]
            o_t = Table(o_rows, colWidths=[5*cm, 2.5*cm, 9.5*cm])
            o_t.setStyle(TABLE_STYLE)
            story.append(o_t)
            story.append(Spacer(1, 0.4*cm))

        # ── Input Parameters ─────────────────────────────────────────
        params = data.get("raw_parameters") or data.get("parameters") or {}
        if params:
            story.append(Paragraph("Input Parameters", h2_style))
            p_data = [[str(k).replace("_"," ").title(), str(v)] for k, v in params.items() if not str(k).startswith("_")]
            if p_data:
                p_table = Table([["Parameter", "Value"]] + p_data, colWidths=[8*cm, 9*cm])
                p_table.setStyle(TABLE_STYLE)
                story.append(p_table)

        # ── SHAP Feature Contributions ───────────────────────────────
        shap = data.get("shap_values", {})
        if shap:
            story.append(Spacer(1, 0.4*cm))
            story.append(Paragraph("Feature Contributions (SHAP)", h2_style))
            s_data = sorted(shap.items(), key=lambda x: abs(float(x[1])) if isinstance(x[1], (int,float)) else 0, reverse=True)
            s_rows = [[str(k).replace("_"," ").title(), _fmt_float(v) if isinstance(v,(int,float)) else str(v)] for k,v in s_data[:10]]
            s_table = Table([["Feature", "Impact Score"]] + s_rows, colWidths=[10*cm, 7*cm])
            s_table.setStyle(TABLE_STYLE)
            story.append(s_table)

        # ── Footer Disclaimer ────────────────────────────────────────
        story.append(Spacer(1, 0.6*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#2d3748")))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "⚠️  Sambhav predictions are probabilistic estimates only and may be incorrect. "
            "Always verify important decisions independently with qualified professionals. "
            "Not a substitute for expert medical, legal, or financial advice.",
            ParagraphStyle("disclaimer", parent=styles["Normal"],
                           textColor=colors.HexColor("#555555"), fontSize=7.5, spaceAfter=0)
        ))

        doc.build(story)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.pdf"}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="ReportLab not installed. Run: pip install reportlab")
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /export/word ─────────────────────────────────────────
@router.post("/word")
async def export_word(req: ExportRequest, db: Session = Depends(get_db)):
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        data = _get_prediction_data(req, db)
        doc = Document()

        # Title
        import datetime
        title = doc.add_heading("Project Sambhav — Prediction Report", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph("Uncertainty, Quantified  ·  A Multi-Modal Probabilistic Inference Engine")
        doc.add_paragraph(f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ·  ID: {data.get('prediction_id', 'N/A')}")

        doc.add_heading("Prediction Summary", 1)
        fp = data.get("final_probability", 0)
        rows_data = [
            ("Prediction ID",    str(data.get("prediction_id", ""))),
            ("Domain",           str(data.get("domain", "")).upper()),
            ("Final Probability", _fmt_pct(fp)),
            ("ML Layer",         _fmt_pct(data.get('ml_probability'))),
            ("LLM Layer",        _fmt_pct(data.get('llm_probability'))),
            ("Reliability Index", _fmt_pct0(data.get('reliability_index'))),
            ("Confidence Tier",  str(data.get("confidence_tier", ""))),
            ("Question",         str(data.get("question", ""))),
        ]
        table = doc.add_table(rows=len(rows_data), cols=2)
        table.style = "Table Grid"
        for i, (label, value) in enumerate(rows_data):
            table.rows[i].cells[0].text = label
            table.rows[i].cells[1].text = value

        # Multi-outcome table
        outcome_items = _get_outcomes_list(data)
        if outcome_items:
            doc.add_heading("Multi-Outcome Predictions", 1)
            o_table = doc.add_table(rows=1 + len(outcome_items), cols=3)
            o_table.style = "Table Grid"
            hdr = o_table.rows[0].cells
            hdr[0].text, hdr[1].text, hdr[2].text = "Outcome", "Probability", "Reasoning"
            for i, (label, prob_pct, reasoning) in enumerate(outcome_items, 1):
                row = o_table.rows[i].cells
                row[0].text = label
                row[1].text = prob_pct
                row[2].text = reasoning[:120] if reasoning else ""

        doc.add_heading("Input Parameters", 1)
        params = data.get("raw_parameters") or data.get("parameters") or {}
        if params:
            p_table = doc.add_table(rows=1, cols=2)
            p_table.style = "Table Grid"
            hdr = p_table.rows[0].cells
            hdr[0].text = "Parameter"
            hdr[1].text = "Value"
            for k, v in params.items():
                if not str(k).startswith("_"):
                    row = p_table.add_row().cells
                    row[0].text = str(k)
                    row[1].text = str(v)

        doc.add_heading("SHAP Feature Contributions", 1)
        shap = data.get("shap_values", {})
        if shap:
            s_table = doc.add_table(rows=1, cols=2)
            s_table.style = "Table Grid"
            hdr = s_table.rows[0].cells
            hdr[0].text = "Feature"
            hdr[1].text = "Contribution"
            for k, v in sorted(shap.items(), key=lambda x: abs(float(x[1])) if isinstance(x[1],(int,float)) else 0, reverse=True)[:10]:
                row = s_table.add_row().cells
                row[0].text = str(k)
                row[1].text = _fmt_float(v) if isinstance(v,(int,float)) else str(v)

        doc.add_paragraph()
        doc.add_paragraph("⚠ Sambhav may be incorrect. Always verify important decisions independently.")

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.docx"}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        logger.error(f"Word export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /export/excel ────────────────────────────────────────
@router.post("/excel")
async def export_excel(req: ExportRequest, db: Session = Depends(get_db)):
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment

        data = _get_prediction_data(req, db)
        wb = openpyxl.Workbook()

        # Sheet 1: Summary
        ws = wb.active
        ws.title = "Summary"
        header_fill = PatternFill("solid", fgColor="1a1a1a")
        header_font = Font(color="c0c0c0", bold=True)
        val_font    = Font(color="e8e8e8")

        rows = [
            ("Field", "Value"),
            ("Prediction ID",   data.get("prediction_id", "")),
            ("Domain",          str(data.get("domain", "")).upper()),
            ("Question",        data.get("question", "")),
            ("Final Probability", _fmt_pct(data.get('final_probability'))),
            ("ML Probability",  _fmt_pct(data.get('ml_probability'))),
            ("LLM Probability", _fmt_pct(data.get('llm_probability'))),
            ("Reliability Index", _fmt_pct0(data.get('reliability_index'))),
            ("Confidence Tier", data.get("confidence_tier", "")),
            ("Mode",            data.get("mode", "")),
        ]
        for r_idx, (field, value) in enumerate(rows, 1):
            ws.cell(r_idx, 1, field).font  = header_font if r_idx == 1 else val_font
            ws.cell(r_idx, 2, str(value)).font = val_font
            if r_idx == 1:
                ws.cell(r_idx, 1).fill = header_fill
                ws.cell(r_idx, 2).fill = header_fill
        ws.column_dimensions["A"].width = 22
        ws.column_dimensions["B"].width = 40

        # Sheet 2: Parameters
        ws2 = wb.create_sheet("Parameters")
        ws2.cell(1, 1, "Parameter").font = header_font
        ws2.cell(1, 2, "Value").font     = header_font
        params = data.get("raw_parameters") or data.get("parameters") or {}
        for i, (k, v) in enumerate(params.items(), 2):
            if not str(k).startswith("_"):
                ws2.cell(i, 1, str(k)).font = val_font
                ws2.cell(i, 2, str(v)).font = val_font

        # Sheet 3: SHAP
        ws3 = wb.create_sheet("SHAP Values")
        ws3.cell(1, 1, "Feature").font     = header_font
        ws3.cell(1, 2, "Contribution").font= header_font
        shap = data.get("shap_values", {})
        for i, (k, v) in enumerate(sorted(shap.items(), key=lambda x: abs(float(x[1])) if isinstance(x[1],(int,float)) else 0, reverse=True), 2):
            ws3.cell(i, 1, str(k)).font  = val_font
            ws3.cell(i, 2, float(v) if isinstance(v,(int,float)) else str(v)).font = val_font

        # Sheet 4: Multi-Outcome Predictions
        ws4 = wb.create_sheet("Outcomes")
        for col_idx, col_header in enumerate(["Outcome", "Probability", "Reasoning"], 1):
            ws4.cell(1, col_idx, col_header).font = header_font
        for i, (label, prob_pct, reasoning) in enumerate(_get_outcomes_list(data), 2):
            ws4.cell(i, 1, label).font    = val_font
            ws4.cell(i, 2, prob_pct).font = val_font
            ws4.cell(i, 3, reasoning).font = val_font
        ws4.column_dimensions["A"].width = 35
        ws4.column_dimensions["B"].width = 15
        ws4.column_dimensions["C"].width = 60

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.xlsx"}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl not installed. Run: pip install openpyxl")
    except Exception as e:
        logger.error(f"Excel export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /export/png ──────────────────────────────────────────
@router.post("/png")
async def export_png(req: ExportRequest, db: Session = Depends(get_db)):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        data = _get_prediction_data(req, db)
        shap = data.get("shap_values", {})

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#08080a")

        # Left: probability gauge
        ax1 = axes[0]
        ax1.set_facecolor("#08080a")
        fp = float(data.get("final_probability") or 0.5)
        theta = np.linspace(np.pi, 0, 200)
        ax1.plot(np.cos(theta), np.sin(theta), color="#1a1a1a", linewidth=20, solid_capstyle="round")
        theta_fill = np.linspace(np.pi, np.pi - fp * np.pi, 200)
        color = "#c0c0c0" if fp >= 0.5 else "#ffb7c5"
        ax1.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=20, solid_capstyle="round")
        ax1.text(0, -0.1, f"{fp*100:.1f}%", ha="center", va="center", fontsize=28, color="#e8e8e8", fontweight="bold")
        ax1.text(0, -0.35, data.get("domain","").upper(), ha="center", va="center", fontsize=11, color="#a0a0a0")
        ax1.text(0, -0.55, f"Reliability: {_fmt_pct0(data.get('reliability_index'))}", ha="center", va="center", fontsize=9, color="#666666")
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-0.8, 1.3)
        ax1.axis("off")
        ax1.set_title("Prediction Probability", color="#c0c0c0", fontsize=12, pad=10)

        # Right: SHAP bar chart
        ax2 = axes[1]
        ax2.set_facecolor("#08080a")
        if shap:
            items = sorted(shap.items(), key=lambda x: abs(float(x[1])) if isinstance(x[1],(int,float)) else 0, reverse=True)[:8]
            features = [str(k)[:20] for k, _ in items]
            values   = [float(v) if isinstance(v,(int,float)) else 0 for _, v in items]
            bar_colors = ["#c0c0c0" if v >= 0 else "#ffb7c5" for v in values]
            bars = ax2.barh(features, values, color=bar_colors, alpha=0.8)
            ax2.axvline(0, color="#333333", linewidth=0.8)
            ax2.tick_params(colors="#a0a0a0", labelsize=9)
            ax2.set_xlabel("SHAP Contribution", color="#a0a0a0", fontsize=9)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#222222")
        ax2.set_title("Feature Contributions (SHAP)", color="#c0c0c0", fontsize=12, pad=10)

        fig.suptitle("Project Sambhav — Prediction Report", color="#e8e8e8", fontsize=14, y=1.02)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#08080a", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=sambhav_{data.get('prediction_id','export')}.png"}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="matplotlib not installed. Run: pip install matplotlib")
    except Exception as e:
        logger.error(f"PNG export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /export/api-link ─────────────────────────────────────
@router.post("/api-link")
async def export_api_link(req: ExportRequest):
    """Returns a shareable read-only API URL for this prediction."""
    pid = req.prediction_id or "demo"
    return JSONResponse({
        "success":      True,
        "api_url":      f"{BASE_URL}/history/{pid}",
        "json_url":     f"{BASE_URL}/export/json",
        "expires_in":   "72 hours",
        "instructions": "POST to api_url with your token to retrieve full prediction data.",
        "curl_example": f"curl -X POST {BASE_URL}/export/json -H 'Content-Type: application/json' -d '{{\"prediction_id\": \"{pid}\"}}'",
    })


# ── Generic dispatch ──────────────────────────────────────────
@router.post("")
@router.post("/{format}")
async def export_dispatch(
    req: ExportRequest,
    format: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Single endpoint that dispatches to format-specific handler."""
    # Prioritize path param over body param
    fmt = (format or req.format or "json").lower()
    dispatch = {
        "json":  export_json,
        "csv":   export_csv,
        "xml":   export_xml,
        "pdf":   export_pdf,
        "word":  export_word,
        "excel": export_excel,
        "png":   export_png,
        "api":   export_api_link,
    }
    handler = dispatch.get(fmt)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Choose from: {list(dispatch.keys())}")
    if fmt == "api":
        return await handler(req)
    return await handler(req, db)