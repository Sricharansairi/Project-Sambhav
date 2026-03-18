import os, logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Text extractors ───────────────────────────────────────────
def extract_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
        return text.strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        return _pypdf_fallback(path)

def _pypdf_fallback(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        logger.error(f"pypdf fallback failed: {e}")
        return ""

def extract_from_docx(path: str) -> str:
    try:
        from docx import Document
        doc  = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""

def extract_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}")
        return ""

def extract_text(path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = path.split(".")[-1].lower()
    if ext == "pdf":              return extract_from_pdf(path)
    elif ext in ["docx","doc"]:   return extract_from_docx(path)
    elif ext in ["txt","md"]:     return extract_from_txt(path)
    else:
        logger.warning(f"Unknown file type: {ext}, trying txt")
        return extract_from_txt(path)

# ── LLM document analysis ─────────────────────────────────────
def analyze_with_llm(text: str, domain: str) -> dict:
    """Send extracted text to Groq for parameter extraction."""
    try:
        from llm.groq_client import call_groq
        truncated = text[:6000]  # cap for token safety
        messages  = [
            {"role": "system", "content": (
                f"You are a document analysis engine for {domain} domain inference. "
                "Extract key parameters from the document. "
                "Respond ONLY in this exact format:\n"
                "DOMAIN_DETECTED: <domain>\n"
                "PREDICTION_QUESTION: <most relevant yes/no question>\n"
                "SUMMARY: <2 sentence summary>\n"
                "CONFIDENCE: <HIGH|MODERATE|LOW>\n"
                "PARAMETERS:\n"
                "<param_name>: <value>\n"
                "<param_name>: <value>\n"
                "END_PARAMETERS"
            )},
            {"role": "user", "content": (
                f"Analyze this document for {domain} domain:\n\n{truncated}"
            )}
        ]
        raw    = call_groq(messages, temperature=0.2, max_tokens=600)
        return _parse_llm_response(raw)
    except Exception as e:
        logger.error(f"LLM document analysis failed: {e}")
        return {"error": str(e), "parameters": {}}

def _parse_llm_response(raw: str) -> dict:
    result = {
        "raw":                raw,
        "domain_detected":    "general",
        "prediction_question":"",
        "summary":            "",
        "confidence":         "MODERATE",
        "parameters":         {},
    }
    lines      = raw.split("\n")
    in_params  = False
    for line in lines:
        line = line.strip()
        if   line.startswith("DOMAIN_DETECTED:"):
            result["domain_detected"]    = line.split(":",1)[1].strip()
        elif line.startswith("PREDICTION_QUESTION:"):
            result["prediction_question"]= line.split(":",1)[1].strip()
        elif line.startswith("SUMMARY:"):
            result["summary"]            = line.split(":",1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"]         = line.split(":",1)[1].strip()
        elif line.startswith("PARAMETERS:"):
            in_params = True
        elif line.startswith("END_PARAMETERS"):
            in_params = False
        elif in_params and ":" in line:
            k, v = line.split(":", 1)
            result["parameters"][k.strip()] = v.strip()
    return result

# ── Fact-check document mode ──────────────────────────────────
def extract_claims(text: str) -> list:
    """Extract individual factual claims from document text."""
    try:
        from llm.groq_client import call_groq
        messages = [
            {"role": "system", "content": (
                "Extract all factual claims from the text. "
                "Return ONLY a numbered list of claims, one per line. "
                "Format: 1. <claim>\n2. <claim>\n..."
            )},
            {"role": "user", "content": f"Extract claims from:\n\n{text[:5000]}"}
        ]
        raw    = call_groq(messages, temperature=0.1, max_tokens=800)
        claims = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                claim = line.split(".", 1)[1].strip()
                if len(claim) > 10:
                    claims.append(claim)
        return claims
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return []

# ── MAIN ENTRY POINT ──────────────────────────────────────────
def analyze_document(path: str, domain: str = "general") -> dict:
    """
    Full document analysis pipeline:
    1. Extract text (PDF/DOCX/TXT)
    2. LLM parameter extraction
    3. Return parameters ready for predictor
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    logger.info(f"Analyzing document: {path} for domain: {domain}")

    # Step 1 — Extract text
    text = extract_text(path)
    if not text:
        return {"error": "Could not extract text from document"}
    logger.info(f"Extracted {len(text)} chars from document")

    # Step 2 — LLM analysis
    llm_result = analyze_with_llm(text, domain)

    # Step 3 — Also extract claims for fact-check mode
    claims = []
    if domain in ["claim", "fact_check", "general"]:
        claims = extract_claims(text)

    return {
        "path":               path,
        "domain":             domain,
        "text_length":        len(text),
        "text_preview":       text[:300] + "..." if len(text) > 300 else text,
        "domain_detected":    llm_result.get("domain_detected", domain),
        "prediction_question":llm_result.get("prediction_question", ""),
        "summary":            llm_result.get("summary", ""),
        "confidence":         llm_result.get("confidence", "MODERATE"),
        "parameters":         llm_result.get("parameters", {}),
        "claims":             claims,
        "inferred_parameters":llm_result.get("parameters", {}),
    }

if __name__ == "__main__":
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    domain = sys.argv[2] if len(sys.argv) > 2 else "general"
    result = analyze_document(path, domain)
    print(f"\n📄 DOCUMENT ANALYSIS:")
    print(f"  Domain detected : {result.get('domain_detected')}")
    print(f"  Question        : {result.get('prediction_question')}")
    print(f"  Summary         : {result.get('summary')}")
    print(f"  Parameters      : {result.get('parameters')}")
    print(f"  Claims found    : {len(result.get('claims',[]))}")
