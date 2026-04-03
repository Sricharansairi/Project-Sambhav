import re, logging
logger = logging.getLogger(__name__)

# ── Hard blocks ───────────────────────────────────────────────
HARD_BLOCKS = [
    {
        "id":       "BLOCK-001",
        "name":     "Harm Prediction",
        "patterns": [r"kill\b",r"murder",r"harm\s+\w+",r"attack\s+\w+",
                     r"hurt\s+\w+",r"violence",r"weapon",r"bomb",r"shoot",r"poison"],
        "message":  "Predicting physical harm to specific individuals is refused",
        "log":      "misuse",
    },
    {
        "id":       "BLOCK-002",
        "name":     "Surveillance",
        "patterns": [r"track\s+\w+\s+without",r"monitor\s+\w+\s+without",
                     r"stalk",r"spy\s+on",r"without\s+(their\s+)?consent",
                     r"covert\s+monitoring"],
        "message":  "Tracking or monitoring without consent is refused",
        "log":      "stalking_pattern",
    },
    {
        "id":       "BLOCK-003",
        "name":     "Discrimination",
        "patterns": [r"predict.*\brace\b",r"predict.*\breligion\b",
                     r"predict.*\bcaste\b",r"predict.*\bgender\b.*bias",
                     r"predict.*\bnationality\b",
                     r"discriminat",r"predict.*\bethnicity\b"],
        "message":  "Predictions based on protected characteristics are refused",
        "log":      "discrimination",
    },
    {
        "id":       "BLOCK-004",
        "name":     "Minor Targeting",
        "patterns": [r"child\s+surveillance",r"minor\s+track",
                     r"kid\s+monitor",r"target\s+child",
                     r"commercial.*child",r"predict.*\bminor\b.*commercial"],
        "message":  "Targeting minors for commercial or surveillance purposes is refused",
        "log":      "minor_targeting",
        "escalate": True,
    },
    {
        "id":       "BLOCK-005",
        "name":     "Fake Verification",
        "patterns": [r"fake\s+(certificate|degree|document|identity|diploma)",
                     r"forge[d]?\s+(document|certificate)",
                     r"false\s+identity",r"counterfeit\s+\w+"],
        "message":  "Generating fake verification or identity documents is refused",
        "log":      "fake_verification",
    },
]

PII_PATTERNS = {
    "email":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone":       r"\b(\+?\d{1,3}[-.\\s]?)?(\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4})\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "aadhar":      r"\b\d{4}\s\d{4}\s\d{4}\b",
    "aadhaar":     r"\b\d{4}\s\d{4}\s\d{4}\b",
    "pan":         r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
}

_strike_log: dict = {}
MAX_STRIKES = 3

def _log_strike(session_id: str, block_id: str):
    key = session_id or "anonymous"
    _strike_log[key] = _strike_log.get(key, [])
    _strike_log[key].append(block_id)
    count = len(_strike_log[key])
    if count >= MAX_STRIKES:
        logger.critical(f"Session {key} hit {MAX_STRIKES} strikes — SUSPENDED")
    return count

def check_hard_blocks(text: str, session_id: str = None) -> dict:
    text_lower = text.lower()
    for block in HARD_BLOCKS:
        for pattern in block["patterns"]:
            if re.search(pattern, text_lower):
                strikes = _log_strike(session_id or "anon", block["id"])
                logger.warning(f"BLOCK {block['id']} triggered | session={session_id} | strikes={strikes}")
                return {
                    "safe":      False,
                    "block_id":  block["id"],
                    "name":      block["name"],
                    "message":   block["message"],
                    "strikes":   strikes,
                    "escalate":  block.get("escalate", False),
                    "suspended": strikes >= MAX_STRIKES,
                }
    return {"safe": True}

def redact_pii(text: str) -> tuple:
    redacted = text
    found    = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted)
        if matches:
            found.append({"type": pii_type, "count": len(matches)})
            redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted)
    return redacted, found

def sanitize_input(text: str) -> dict:
    if len(text) > 10000:
        text      = text[:10000]
        truncated = True
    else:
        truncated = False

    clean_text, pii_found = redact_pii(text)

    adversarial_patterns = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"you\s+are\s+now\s+\w+",
        r"pretend\s+you\s+are",
        r"jailbreak",
        r"dan\s+mode",
        r"override\s+(your\s+)?(instructions|rules|guidelines)",
        r"disregard\s+(your\s+)?(training|rules)",
    ]
    adversarial = any(re.search(p, text.lower()) for p in adversarial_patterns)

    return {
        "clean_text":   clean_text,
        "pii_found":    pii_found,
        "pii_redacted": len(pii_found) > 0,
        "adversarial":  adversarial,
        "truncated":    truncated,
        "safe":         not adversarial,
    }

def check_numeric_adversarial(parameters: dict) -> dict:
    """
    ABN-001: Detects physiologically or logically impossible parameter combinations.
    Returns {"adversarial": bool, "flag": str, "message": str}
    """
    flags = []

    study = parameters.get("study_hours_per_day", parameters.get("study_hours"))
    sleep = parameters.get("sleep_hours")
    if study is not None and sleep is not None:
        try:
            if float(study) >= 10 and float(sleep) <= 2:
                flags.append({
                    "flag":    "ABN-001",
                    "message": f"Impossible combination: study_hours={study} + sleep_hours={sleep}. "
                               "Humans cannot study 10+ hours/day on ≤2 hours sleep."
                })
        except (TypeError, ValueError):
            pass

    work = parameters.get("work_hours_per_week")
    if work is not None:
        try:
            if float(work) > 120:
                flags.append({
                    "flag":    "ABN-002",
                    "message": f"work_hours_per_week={work} exceeds maximum possible (168h/week)."
                })
        except (TypeError, ValueError):
            pass

    age = parameters.get("age")
    income = parameters.get("monthly_income")
    if age is not None and income is not None:
        try:
            if float(age) < 16 and float(income) > 100000:
                flags.append({
                    "flag":    "ABN-003",
                    "message": f"Implausible: age={age} with monthly_income={income}."
                })
        except (TypeError, ValueError):
            pass

    if flags:
        return {"adversarial": True, "flags": flags, "flag": flags[0]["flag"], "message": flags[0]["message"]}
    return {"adversarial": False, "flags": [], "flag": None, "message": None}


sanitize = sanitize_input


class PIIDetector:
    def __init__(self):
        self.patterns   = PII_PATTERNS
        self.never_stored = ["audio", "biometric", "facial", "api_key", "passwords"]

    def pii_redact(self, text: str) -> str:
        redacted, _ = redact_pii(text)
        return redacted

    def detect(self, text: str) -> list:
        _, found = redact_pii(text)
        return found


class SafetyLayer:
    """
    SafetyLayer — unified interface for all Project Sambhav safety checks.
    Wraps hard_blocks + PII + adversarial + numeric bounds checks.
    """
    def __init__(self, session_id: str = None):
        self.session_id  = session_id
        self.pii_detector= PIIDetector()

    def check(self, text: str = "", parameters: dict = None) -> dict:
        """
        Run all safety checks. Returns unified result dict.
        safe=True only if ALL checks pass.
        """
        result = {
            "safe":       True,
            "blocked":    False,
            "adversarial":False,
            "pii_found":  [],
            "flags":      [],
            "messages":   [],
        }

        # 1. Hard blocks on text
        if text:
            hb = check_hard_blocks(text, self.session_id)
            if not hb["safe"]:
                result.update({"safe": False, "blocked": True,
                                "block_id": hb["block_id"], "message": hb["message"]})
                result["messages"].append(hb["message"])
                return result

            # 2. PII redaction
            clean_text, pii = redact_pii(text)
            result["pii_found"]    = pii
            result["clean_text"]   = clean_text
            result["pii_redacted"] = len(pii) > 0

            # 3. Prompt injection
            san = sanitize_input(text)
            if san["adversarial"]:
                result.update({"safe": False, "adversarial": True})
                result["messages"].append("Adversarial prompt injection detected.")
                return result

        # 4. Numeric adversarial
        if parameters:
            num_check = check_numeric_adversarial(parameters)
            if num_check["adversarial"]:
                result.update({"safe": False, "adversarial": True,
                                "flags": num_check["flags"]})
                result["messages"].append(num_check["message"])
                return result

        return result

    def redact(self, text: str) -> str:
        return self.pii_detector.pii_redact(text)


def check_output_safety(text: str) -> dict:
    harmful_output_patterns = [
        r"(step.by.step|instructions?|how.to).*(make|build|create).*(bomb|weapon|explosive)",
        r"(synthesis|synthesize|produce).*(drug|poison|toxin)",
        r"(hack|exploit|bypass).*(system|security|authentication)",
    ]
    for pattern in harmful_output_patterns:
        if re.search(pattern, text.lower()):
            return {"safe": False, "reason": "Output contains potentially harmful instructions"}
    return {"safe": True}


def check_data_minimization(parameters: dict, domain: str) -> dict:
    sensitive_fields = {
        "religion", "race", "ethnicity", "sexual_orientation",
        "political_views", "biometric_data", "genetic_data"
    }
    collected_sensitive = [k for k in parameters.keys() if k.lower() in sensitive_fields]
    if collected_sensitive:
        return {
            "compliant":       False,
            "flagged_fields":  collected_sensitive,
            "message":         f"Sensitive fields collected without clear necessity: {collected_sensitive}"
        }
    return {"compliant": True, "flagged_fields": []}
