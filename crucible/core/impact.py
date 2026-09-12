"""Quantified impact, scale metrics, and action-verb seniority scoring."""

import re
from dataclasses import dataclass, field


@dataclass
class ImpactAnalysis:
    impact_score: float  # 0.0 - 1.0
    metrics_count: int
    leadership_verbs_count: int
    detected_metrics: list[str] = field(default_factory=list)
    detected_verbs: list[str] = field(default_factory=list)


# High-seniority / high-impact action verbs
STRONG_ACTION_VERBS: set[str] = {
    "architected",
    "spearheaded",
    "orchestrated",
    "optimized",
    "scaled",
    "engineered",
    "designed",
    "overhauled",
    "streamlined",
    "mentored",
    "led",
    "built",
    "deployed",
    "migrated",
    "automated",
    "reduced",
    "increased",
}

# Regex patterns for business impact and quantitative achievements
METRIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d+(?:\.\d+)?%(?!\w)"),  # Percentages (e.g. 40%, 15.5%)
    re.compile(r"\$\s*\d+(?:\.\d+)?[kKmMbB]?\b"),  # Dollar amounts ($50k, $1.2M)
    re.compile(r"\b\d+(?:\.\d+)?x\b", re.I),  # Multipliers (e.g. 10x speedup)
    re.compile(
        r"\b\d+\+?\s*(?:k|m|million|thousand)?\s*(?:users|rps|qps|requests|events|tps|records|customers|clients)\b",
        re.I,
    ),  # Scale
    re.compile(r"\b(?:reduced|saved|cut)\s+\w+\s+by\s+\d+", re.I),  # Reductions
    re.compile(r"\b(?:increased|improved|boosted)\s+\w+\s+by\s+\d+", re.I),  # Boosts
]


def analyze_impact(text: str) -> ImpactAnalysis:
    """Extract quantified metrics and strong leadership verbs to calculate impact score."""
    if not text:
        return ImpactAnalysis(
            impact_score=0.0,
            metrics_count=0,
            leadership_verbs_count=0,
            detected_metrics=[],
            detected_verbs=[],
        )

    # 1. Detect quantitative metrics
    detected_metrics: list[str] = []
    for pattern in METRIC_PATTERNS:
        for match in pattern.finditer(text):
            cleaned = match.group(0).strip()
            if cleaned and cleaned not in detected_metrics:
                detected_metrics.append(cleaned)

    # 2. Detect strong leadership verbs
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    detected_verbs = sorted({w for w in words if w in STRONG_ACTION_VERBS})

    metrics_count = len(detected_metrics)
    verbs_count = len(detected_verbs)

    # Scoring formula: 5 metrics or 6 strong verbs maxes out impact score
    metric_subscore = min(metrics_count / 4.0, 1.0)
    verb_subscore = min(verbs_count / 5.0, 1.0)

    # Weighted blend: metrics carry 60%, action verbs carry 40%
    score = (0.60 * metric_subscore) + (0.40 * verb_subscore)

    return ImpactAnalysis(
        impact_score=round(score, 4),
        metrics_count=metrics_count,
        leadership_verbs_count=verbs_count,
        detected_metrics=detected_metrics[:8],
        detected_verbs=detected_verbs[:8],
    )
