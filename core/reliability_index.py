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
    """
    Compute Reliability Index — 0.0 to 1.0
    Returns full breakdown not just score.
    """
    skipped  = skipped or []
    breakdown = {}

    # ── Component 1: Parameter completeness (40% weight) ─────
    provided = len([v for v in parameters.values() if v is not None])
    total    = max(provided + len(skipped), 1)
    param_score = provided / total
    breakdown["parameter_completeness"] = round(param_score, 3)

    # ── Component 2: Layer availability (30% weight) ──────────
    layer_score = 0.0
    if ml_available  and llm_available:  layer_score = 1.0
    elif ml_available or llm_available:  layer_score = 0.6
    else:                                layer_score = 0.1
    breakdown["layer_availability"] = round(layer_score, 3)

    # ── Component 3: Gap penalty (20% weight) ─────────────────
    if   gap < 0.10: gap_score = 1.0
    elif gap < 0.25: gap_score = 0.8
    elif gap < 0.40: gap_score = 0.5
    else:            gap_score = 0.2
    breakdown["gap_score"] = round(gap_score, 3)

    # ── Component 4: Vision bonus (10% weight) ────────────────
    vision_score = 0.5  # baseline
    if vision_used:
        vision_score = min(1.0, 0.7 + frames_analyzed * 0.01)
    breakdown["vision_score"] = round(vision_score, 3)

    # ── Weighted final score ──────────────────────────────────
    final = (
        param_score  * 0.40 +
        layer_score  * 0.30 +
        gap_score    * 0.20 +
        vision_score * 0.10
    )

    # Skipped high-weight params penalty
    penalty = len(skipped) * 0.04
    final   = max(0.10, min(1.0, final - penalty))

    # ── Confidence tier label ─────────────────────────────────
    if   final >= 0.85: tier = "HIGH"
    elif final >= 0.65: tier = "MODERATE"
    elif final >= 0.40: tier = "LOW"
    else:               tier = "VERY_LOW"

    return {
        "score":      round(final, 3),
        "tier":       tier,
        "breakdown":  breakdown,
        "skipped":    skipped,
        "pct":        f"{final*100:.0f}%",
    }

def display_color(score: float) -> str:
    """Return sage-lime color hex based on score."""
    if   score >= 0.85: return "#C2CD93"  # sage-lime full
    elif score >= 0.65: return "#787858"  # sage-lime dim
    elif score >= 0.40: return "#4B5234"  # sage-lime fade
    else:               return "#E74C3C"  # error red

if __name__ == "__main__":
    result = compute(
        parameters={"study_hours":3,"attendance":75,"stress":"high"},
        domain="student",
        skipped=["motivation"],
        ml_available=True,
        llm_available=True,
        gap=0.15,
    )
    print(f"\n  Reliability Score : {result['pct']}")
    print(f"  Tier              : {result['tier']}")
    print(f"  Breakdown         : {result['breakdown']}")
