import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.units import cm

def generate_pdf(data, output_path, tier="Detailed"):
    """
    Professional PDF generator for Project Sambhav.
    Includes: Multi-layer probabilities, SHAP breakdown, and Audit status.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=1.5*cm, rightMargin=1.5*cm)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=24, textColor=colors.HexColor("#2C3E50"), spaceAfter=10)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#7F8C8D"), spaceAfter=20)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=16, textColor=colors.HexColor("#2980B9"), spaceBefore=15, spaceAfter=10)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, spaceAfter=6)
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=10, fontName="Helvetica-Bold")
    
    story = []

    # 1. Header
    story.append(Paragraph(data.get("project_name", "Project Sambhav"), title_style))
    story.append(Paragraph(f"{data.get('subtitle', '')} — {data.get('version', 'v1.1')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=20))

    # 2. Summary Info
    summary_data = [
        [Paragraph("Prediction ID:", label_style), data.get("prediction_id", "N/A"), 
         Paragraph("Domain:", label_style), data.get("domain", "N/A")],
        [Paragraph("Generated At:", label_style), data.get("generated_at", "N/A"),
         Paragraph("Mode:", label_style), data.get("mode", "guided").title()]
    ]
    t = Table(summary_data, colWidths=[4*cm, 5*cm, 3*cm, 6*cm])
    t.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'LEFT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t)
    story.append(Spacer(1, 1*cm))

    # 3. Probabilistic Analysis (The "Multi Probs" requested)
    story.append(Paragraph("Probabilistic Inference Layers", h2_style))
    prob_data = [
        ["Layer", "Probability", "Contribution", "Status"],
        ["ML Prediction Layer", data.get("ml_probability", "N/A"), "65%", "CALIBRATED"],
        ["LLM Inference Layer", data.get("llm_probability", "N/A"), "35%", "STOCHASTIC"],
        ["Reconciled Final", data.get("reconciled_probability", "N/A"), "100%", "OPTIMIZED"]
    ]
    pt = Table(prob_data, colWidths=[5*cm, 4*cm, 4*cm, 5*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ECF0F1")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#D5F5E3")) # Final row highlight
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(f"<b>Reliability Index:</b> {data.get('reliability_index', 'N/A')}% ({data.get('warning_level', 'CLEAR')})", body_style))
    story.append(Paragraph(f"<b>Agreement Gap:</b> {data.get('agreement_gap', 'N/A')} | <b>Confidence Tier:</b> {data.get('confidence_tier', 'N/A')}", body_style))

    # 4. Outcomes
    story.append(Paragraph("Detailed Outcomes", h2_style))
    outcomes = data.get("outcomes", [])
    if outcomes:
        out_data = [["Outcome Label", "Probability Score"]]
        for o in outcomes:
            out_data.append([o.get("label", ""), o.get("probability", "")])
        ot = Table(out_data, colWidths=[9*cm, 9*cm])
        ot.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F8F9F9")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT')
        ]))
        story.append(ot)
    else:
        story.append(Paragraph("No specific outcomes generated.", body_style))

    # 5. Parameters (Inputs)
    story.append(Paragraph("Input Parameters", h2_style))
    params = data.get("parameters", {})
    if params:
        param_data = [["Parameter", "Value"]]
        for k, v in params.items():
            param_data.append([k.replace("_", " ").title(), str(v)])
        p_table = Table(param_data, colWidths=[9*cm, 9*cm])
        p_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT')
        ]))
        story.append(p_table)

    # 6. SHAP Values (Feature Importance)
    story.append(Paragraph("Feature Contribution (SHAP)", h2_style))
    shaps = data.get("shap_values", {})
    if shaps:
        shap_data = [["Feature", "Impact Score", "Direction"]]
        for k, v in shaps.items():
            direction = "Positive (+)" if v > 0 else "Negative (-)"
            shap_data.append([k.replace("_", " ").title(), f"{v:+.4f}", direction])
        s_table = Table(shap_data, colWidths=[6*cm, 6*cm, 6*cm])
        s_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER')
        ]))
        story.append(s_table)
    else:
        story.append(Paragraph("Feature importance analysis not available for this mode.", body_style))

    # 7. Audit & Safety
    story.append(Paragraph("Audit & Safety Compliance", h2_style))
    audit = data.get("audit", {})
    story.append(Paragraph(f"<b>Overall Status:</b> {audit.get('overall_status', 'PASSED')}", body_style))
    flags = audit.get("flags", [])
    if flags:
        for f in flags:
            story.append(Paragraph(f"• [{f.get('code', 'INFO')}] {f.get('message', '')} ({f.get('severity', 'INFO')})", body_style))
    else:
        story.append(Paragraph("• All safety and integrity checks passed.", body_style))

    # Footer
    story.append(Spacer(1, 2*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(data.get("disclaimer", ""), body_style))
    story.append(Paragraph("© 2026 Sricharan Sairi. Generated via Project Sambhav Inference Engine.", subtitle_style))

    doc.build(story)
