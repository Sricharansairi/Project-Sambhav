import logging
logger = logging.getLogger(__name__)


def compute(
    parameters:      dict,
    domain:          str,
    skipped:         list = None,
    vision_used:     bool = False,
    frames_analyzed: int  = 0,
    ml_available:    bool = True,
    llm_available:   bool = True,
    gap:             float= 0.0,
) -> dict:
    skipped   = skipped or []
    breakdown = {}

    provided    = len([v for v in parameters.values() if v is not None])
    total       = max(provided + len(skipped), 1)
    param_score = provided / total
    breakdown["parameter_completeness"] = round(param_score, 3)

    layer_score = 0.0
    if ml_available  and llm_available:  layer_score = 1.0
    elif ml_available or llm_available:  layer_score = 0.6
    else:                                layer_score = 0.1
    breakdown["layer_availability"] = round(layer_score, 3)

    if   gap < 0.10: gap_score = 1.0
    elif gap < 0.25: gap_score = 0.8
    elif gap < 0.40: gap_score = 0.5
    else:            gap_score = 0.2
    breakdown["gap_score"] = round(gap_score, 3)

    vision_score = 0.5
    if vision_used:
        vision_score = min(1.0, 0.7 + frames_analyzed * 0.01)
    breakdown["vision_score"] = round(vision_score, 3)

    final   = (param_score * 0.40 + layer_score * 0.30 +
               gap_score   * 0.20 + vision_score * 0.10)
    penalty = len(skipped) * 0.04
    final   = max(0.10, min(1.0, final - penalty))

    if   final >= 0.75: tier = "CLEAR"
    elif final >= 0.50: tier = "MODERATE"
    elif final >= 0.30: tier = "LOW"
    else:               tier = "CRITICAL"

    return {
        "score":      round(final, 3),
        "tier":       tier,
        "breakdown":  breakdown,
        "skipped":    skipped,
        "pct":        f"{(final or 0.0)*100:.0f}%",
        "suggestions": "SHAP-powered improvement suggestions apply here to increase reliability."
    }


def display_color(score: float) -> str:
    if   score >= 0.85: return "#C2CD93"
    elif score >= 0.65: return "#787858"
    elif score >= 0.40: return "#4B5234"
    else:               return "#E74C3C"


class ReliabilityIndex:
    """
    ReliabilityIndex — class interface for the reliability computation module.
    Wraps the compute() function with instance state for repeat queries.
    """
    def __init__(self, domain: str = "", skipped: list = None):
        self.domain  = domain
        self.skipped = skipped or []
        self._last   = None

    def compute(self, parameters: dict, ml_available: bool = True,
                llm_available: bool = True, gap: float = 0.0,
                vision_used: bool = False, frames_analyzed: int = 0) -> dict:
        result = compute(
            parameters      = parameters,
            domain          = self.domain,
            skipped         = self.skipped,
            vision_used     = vision_used,
            frames_analyzed = frames_analyzed,
            ml_available    = ml_available,
            llm_available   = llm_available,
            gap             = gap,
        )
        self._last = result
        return result

    def score(self) -> float:
        return self._last["score"] if self._last else 0.0

    def tier(self) -> str:
        return self._last["tier"] if self._last else "UNKNOWN"

    def color(self) -> str:
        return display_color(self.score())

    @staticmethod
    def from_dict(data: dict) -> "ReliabilityIndex":
        ri = ReliabilityIndex(domain=data.get("domain", ""))
        ri._last = data
        return ri
