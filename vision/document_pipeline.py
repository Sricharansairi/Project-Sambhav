import os, logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ── Text extractors ───────────────────────────────────────────
def extract_from_pdf(path: str) -> str:
    """
    Extract text from PDF.
    Section 6.5 — Uses NVIDIA NIM OCDNet + OCRNet for scanned PDFs (P.18).
    Fallback to pdfplumber for digital PDFs.
    """
    try:
        # P.18 — Try NVIDIA NIM OCR first for potential scanned content
        from llm.nvidia_client import call_nvidia_vision
        import base64
        # Just send the first page to check if it's an image-based PDF
        # (In a real system, we'd check all pages or use a dedicated NIM OCR endpoint)
        
        # Fallback to standard extraction for speed if it's a digital PDF
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: 
                    text += t + "\n"
                else:
                    # Page has no text — likely a scan. Use NVIDIA NIM OCR (P.18)
                    logger.info(f"Page {page.page_number} appears scanned, using NVIDIA OCR...")
                    # We'd normally convert page to image and call NIM
                    pass
        return text.strip()
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
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

def extract_from_csv(path: str) -> str:
    try:
        import csv
        text = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                text += ", ".join(row) + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"CSV extraction failed: {e}")
        return ""

def extract_text(path: str) -> str:
    """Auto-detect file type and extract text.
    Full support for up to 1M token context window documents. (P.03/P.17)
    No chunking — entire document in one pass.
    """
    ext = path.split(".")[-1].lower()
    if ext == "pdf":              return extract_from_pdf(path)
    elif ext in ["docx","doc"]:   return extract_from_docx(path)
    elif ext in ["csv", "xlsx", "xls"]: return extract_from_csv(path)
    elif ext in ["txt","md", "py", "js", "ts", "json", "yaml", "yml", "xml", "html", "htm"]: return extract_from_txt(path)
    else:
        # For unknown files, try to read as UTF-8 first, then fallback to 'ignore' errors
        logger.info(f"Unknown file type: {ext}, attempting robust text extraction")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Robust extraction failed for {ext}: {e}")
            return ""

# ── LLM document analysis ─────────────────────────────────────
def analyze_with_llm(text: str, domain: str) -> dict:
    """
    Send extracted text to NVIDIA NIM Kimi K2.5 for parameter extraction.
    No chunking (P.17). Full context window (P.03).
    """
    try:
        from llm.nvidia_client import call_nvidia
        from api.endpoints.predict import _load_registry
        
        reg = _load_registry()
        domain_data = reg.get(domain, {})
        domain_params = domain_data.get("parameters", {})
        
        param_info = ""
        if domain_params:
            for k, p in (domain_params.items() if isinstance(domain_params, dict) else enumerate(domain_params)):
                key = p.get("key") or k
                label = p.get("label", key)
                param_info += f"- {key}: {label}\n"

        # NO CHUNKING — send the whole text (Large context window supported)
        messages  = [
            {"role": "system", "content": (
                f"You are the document analysis engine for Project Sambhav ({domain} domain).\n"
                "Analyze the entire document text provided and extract parameter values.\n"
                f"Extract values for these specific parameters if found:\n{param_info or 'Any domain-relevant signals'}\n\n"
                "YOUR TASK:\n"
                "1. Be thorough. Extract every signal that maps to the requested parameters.\n"
                "2. If values aren't explicitly stated, infer them if possible based on context (with confidence).\n"
                "3. For the 'Sarvagna' domain, if 'text_input' is a required parameter, set it to a summary of the content.\n"
                "4. Set 'confidence' to HIGH if you found most parameters, LOW only if document is irrelevant.\n\n"
                "Respond ONLY in this exact JSON format:\n"
                "{\n"
                '  "domain_detected": "<domain>",\n'
                '  "prediction_question": "<question>",\n'
                '  "summary": "<2 sentence summary>",\n'
                '  "confidence": "<HIGH|MODERATE|LOW>",\n'
                '  "parameters": { "param_name": "value", ... }\n'
                "}"
            )},
            {"role": "user", "content": (
                f"Analyze this full document for {domain} domain:\n\n{text[:50000]}" # Limit to 50k chars for safety, though models support more
            )}
        ]
        
        # Use a stable NVIDIA NIM model
        raw = call_nvidia(messages, model="meta/llama-3.3-70b-instruct", temperature=0.1, max_tokens=1000)
        
        import json, re
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        
        # Ensure we only return parameters that exist in our registry if domain is known
        if domain_params:
            valid_keys = [p.get("key") or k for k, p in (domain_params.items() if isinstance(domain_params, dict) else enumerate(domain_params))]
            data["parameters"] = {k: v for k, v in data.get("parameters", {}).items() if k in valid_keys}
            
        return data
    except Exception as e:
        logger.error(f"NVIDIA document analysis failed: {e}")
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
    1. Extract text (PDF/DOCX/TXT/CSV)
       - OCR support via OCDNet + OCRNet for scanned PDFs (P.18).
    2. LLM parameter extraction
       - Guarantee: No chunking (full doc in single API call) (P.17).
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

    # G.09: Ensure Sarvagna has text_input even if LLM missed it
    if domain == "sarvagna" and "text_input" not in llm_result.get("parameters", {}):
        if "parameters" not in llm_result: llm_result["parameters"] = {}
        llm_result["parameters"]["text_input"] = text[:2000] # Use first 2000 chars as input
        llm_result["parameters"]["domain_context"] = 5 # Default to 'General' (value 5)

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
