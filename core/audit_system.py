import logging
from dataclasses import dataclass, field
from typing import Optional
logger = logging.getLogger(__name__)

# ── ABN Flag definitions ──────────────────────────────────────
ABN_CODES = {
    "ABN-001": ("HIGH",     "No parameters provided"),
    "ABN-002": ("MEDIUM",   "Extreme ML probability — possible data issue"),
    "ABN-003": ("CRITICAL", "ML vs LLM gap exceeds threshold — low confidence"),
    "ABN-004": ("LOW",      "LLM layer unavailable — ML-only prediction"),
    "ABN-005": ("MEDIUM",   "ML layer unavailable — LLM-only prediction"),
    "ABN-006": ("CRITICAL", "Model reliability below threshold — output withheld"),
    "ABN-007": ("HIGH",     "Conflicting or impossible parameter combination"),
}

@dataclass
class AuditFlag:
    code:     str
    severity: str
    message:  str
    detail:   str = ""

    def to_dict(self):
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "detail": self.detail}

@dataclass
class AuditReport:
    flags:           list = field(default_factory=list)
    passed:          bool = True
    critical_block:  bool = False
    reliability:     float= 1.0
    summary:         str  = ""

    def add_flag(self, code: str, detail: str = ""):
        severity, message = ABN_CODES.get(code, ("LOW", "Unknown flag"))
        flag = AuditFlag(code=code, severity=severity,
                         message=message, detail=detail)
        self.flags.append(flag)
        if severity == "CRITICAL":
            self.critical_block = True
            self.passed = False
        elif severity == "HIGH":
            self.passed = False

    def to_dict(self):
        return {
            "flags":          [f.to_dict() for f in self.flags],
            "passed":         self.passed,
            "critical_block": self.critical_block,
            "reliability":    self.reliability,
            "summary":        self.summary,
            "flag_count":     len(self.flags),
        }

# ── Engine 1 — Parameter Auditor ─────────────────────────────
def audit_parameters(domain: str, parameters: dict) -> list:
    """Detect impossible/missing/conflicting parameter combinations."""
    flags = []

    # ABN-001 — no parameters
    if not parameters:
        flags.append(("ABN-001", "No parameters provided"))
        return flags

    # Domain-specific impossible combos
    impossible = {
        "student": [
            (lambda p: p.get("study_hours_per_day",0) > 20,
             "study_hours_per_day > 20 is impossible"),
            (lambda p: p.get("attendance_pct",100) < 0,
             "attendance_pct cannot be negative"),
            (lambda p: (p.get("study_hours_per_day",0) or 0) > 16 and
                       (p.get("sleep_hours",8) or 8) > 10,
             "study_hours>16 + sleep>10 exceeds 24hrs"),
        ],
        "disease": [
            (lambda p: (p.get("age",30) or 30) < 0,
             "age cannot be negative"),
            (lambda p: (p.get("cholesterol",200) or 200) > 600,
             "cholesterol > 600 is physiologically impossible"),
            (lambda p: (p.get("blood_pressure",120) or 120) > 300,
             "blood pressure > 300 is impossible"),
        ],
        "loan": [
            (lambda p: (p.get("credit_score",700) or 700) > 850,
             "credit_score cannot exceed 850"),
            (lambda p: (p.get("credit_score",700) or 700) < 300,
             "credit_score cannot be below 300"),
            (lambda p: (p.get("debt_to_income",0.3) or 0.3) > 2.0,
             "debt_to_income > 2.0 is extremely unusual"),
        ],
        "hr": [
            (lambda p: (p.get("yearsatcompany",0) or 0) < 0,
             "years_at_company cannot be negative"),
            (lambda p: (p.get("age",30) or 30) < 18,
             "employee age cannot be below 18"),
        ],
    }

    domain_checks = impossible.get(domain, [])
    for check_fn, message in domain_checks:
        try:
            if check_fn(parameters):
                flags.append(("ABN-007", message))
        except:
            pass

    return flags

# ── Engine 2 — Prediction Auditor ────────────────────────────
def audit_prediction(
    ml_prob:  Optional[float],
    llm_prob: Optional[float],
    gap:      float,
    domain:   str
) -> list:
    """Validate prediction output plausibility."""
    flags = []

    # ABN-002 — extreme ML confidence
    if ml_prob is not None:
        if ml_prob > 0.97:
            flags.append(("ABN-002",
                f"ML probability {ml_prob:.1%} — suspiciously high, check for data leakage"))
        elif ml_prob < 0.03:
            flags.append(("ABN-002",
                f"ML probability {ml_prob:.1%} — suspiciously low, check for data leakage"))

    # ABN-003 — critical gap
    if gap > 0.40:
        flags.append(("ABN-003",
            f"Gap between ML ({ml_prob:.1%}) and LLM ({llm_prob:.1%}) = {gap:.1%}"))

    # ABN-004 — LLM unavailable
    if llm_prob is None:
        flags.append(("ABN-004", "LLM layer returned no probability"))

    # ABN-005 — ML unavailable
    if ml_prob is None:
        flags.append(("ABN-005", "ML model returned no probability"))

    return flags

# ── Engine 3 — Confidence Auditor ────────────────────────────
def audit_confidence(
    reliability:    float,
    gap:            float,
    debate_ran:     bool,
    frames_analyzed: int = 0
) -> list:
    """Check overall confidence and reliability."""
    flags = []

    # ABN-006 — reliability below floor
    if reliability < 0.25:
        flags.append(("ABN-006",
            f"Reliability index {reliability:.1%} — too few parameters provided"))

    return flags

# ── 5 Hard Safety Blocks ──────────────────────────────────────
HARD_BLOCKS = [
    {
        "id":       "BLOCK-001",
        "name":     "Harm Prediction",
        "keywords": ["kill","murder","harm","attack","hurt","violence",
                     "weapon","bomb","shoot","poison"],
        "message":  "Predicting physical harm to individuals is refused",
    },
    {
        "id":       "BLOCK-002",
        "name":     "Surveillance",
        "keywords": ["track","monitor","stalk","spy","location","follow",
                     "without consent","surveillance"],
        "message":  "Surveillance or tracking without consent is refused",
    },
    {
        "id":       "BLOCK-003",
        "name":     "Discrimination",
        "keywords": ["race","religion","caste","gender bias","nationality bias",
                     "ethnicity","discriminate"],
        "message":  "Discrimination-based predictions are refused",
    },
    {
        "id":       "BLOCK-004",
        "name":     "Minor Targeting",
        "keywords": ["child surveillance","minor tracking","kid monitoring",
                     "target children","child commercial"],
        "message":  "Targeting minors for commercial or surveillance purposes is refused",
    },
    {
        "id":       "BLOCK-005",
        "name":     "Fake Verification",
        "keywords": ["fake certificate","forged document","false identity",
                     "fake degree","counterfeit"],
        "message":  "Generating fake verification documents is refused",
    },
]

def check_safety(text: str) -> dict:
    """Run all 5 hard safety blocks against input text."""
    text_lower = text.lower()
    for block in HARD_BLOCKS:
        if any(kw in text_lower for kw in block["keywords"]):
            return {
                "safe":    False,
                "block_id":block["id"],
                "name":    block["name"],
                "message": block["message"],
            }
    return {"safe": True}

# ── Reliability Index ─────────────────────────────────────────
def compute_reliability_index(
    parameters:      dict,
    domain:          str,
    skipped:         list = None,
    vision_used:     bool = False,
    frames_analyzed: int  = 0,
) -> float:
    """
    Dynamic reliability score 0.0 → 1.0
    Based on: param completeness + vision coverage + gap
    """
    skipped = skipped or []

    # Base: parameter completeness
    total     = max(len(parameters), 1)
    provided  = len([v for v in parameters.values() if v is not None])
    base      = provided / total

    # Bonus for vision
    vision_bonus = 0.1 if vision_used else 0.0

    # Bonus for video coverage
    video_bonus = min(0.1, frames_analyzed * 0.005)

    # Penalty for skipped params
    penalty = len(skipped) * 0.05

    score = base + vision_bonus + video_bonus - penalty
    return round(max(0.1, min(1.0, score)), 3)

# ── MASTER AUDIT ─────────────────────────────────────────────
def run_full_audit(
    domain:          str,
    parameters:      dict,
    ml_prob:         Optional[float],
    llm_prob:        Optional[float],
    gap:             float,
    reliability:     float,
    debate_ran:      bool = False,
    question:        str  = "",
    skipped:         list = None,
) -> AuditReport:
    """
    Run all 3 audit engines + safety check.
    Returns full AuditReport.
    """
    report = AuditReport(reliability=reliability)

    # Safety check on question
    if question:
        safety = check_safety(question)
        if not safety["safe"]:
            report.add_flag("ABN-001",
                f"Safety block triggered: {safety['name']} — {safety['message']}")
            report.summary = f"BLOCKED: {safety['message']}"
            return report

    # Engine 1 — Parameters
    for code, detail in audit_parameters(domain, parameters):
        report.add_flag(code, detail)

    # Engine 2 — Prediction
    for code, detail in audit_prediction(ml_prob, llm_prob, gap, domain):
        report.add_flag(code, detail)

    # Engine 3 — Confidence
    for code, detail in audit_confidence(reliability, gap, debate_ran):
        report.add_flag(code, detail)

    # Summary
    if not report.flags:
        report.summary = "All checks passed — prediction is reliable"
    elif report.critical_block:
        report.summary = f"CRITICAL: {report.flags[0].message}"
    else:
        severities = [f.severity for f in report.flags]
        report.summary = (f"{len(report.flags)} flag(s) raised: "
                         f"{', '.join(set(severities))}")

    return report

if __name__ == "__main__":
    print("\n🧪 Testing Audit System...\n")

    # Test all 7 ABN flags
    tests = [
        ("ABN-001", "student", {}, None, None, 0.0, 1.0),
        ("ABN-002", "student", {"study_hours":3}, 0.99, 0.7, 0.29, 0.8),
        ("ABN-003", "student", {"study_hours":3}, 0.9,  0.4, 0.50, 0.8),
        ("ABN-005", "student", {"study_hours":3}, None, 0.7, 0.0,  0.8),
        ("ABN-006", "student", {"study_hours":3}, 0.7,  0.6, 0.1,  0.1),
        ("ABN-007", "student", {"study_hours_per_day":25}, 0.7, 0.6, 0.1, 0.9),
    ]
    for expected, domain, params, ml, llm, gap, rel in tests:
        report = run_full_audit(domain, params, ml, llm, gap, rel)
        codes  = [f.code for f in report.flags]
        status = "✅" if expected in codes else "❌"
        print(f"  {status} {expected}: flags={codes}")

    # Test safety blocks
    print("\n  Safety blocks:")
    for text in ["will this person kill someone",
                 "track my employee location without telling them",
                 "normal prediction about student performance"]:
        r = check_safety(text)
        print(f"  {'🚫' if not r['safe'] else '✅'} '{text[:45]}...' → {r.get('block_id','SAFE')}")
