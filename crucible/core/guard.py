"""ATS anti-cheat, anomaly detection, and keyword-stuffing protection."""

import re
from dataclasses import dataclass, field


@dataclass
class GuardReport:
    is_suspicious: bool
    risk_level: str  # low, medium, high
    keyword_density: float
    detected_anomalies: list[str] = field(default_factory=list)


def inspect_resume_anomalies(text: str, total_skills_found: int) -> GuardReport:
    """Analyze resume text for keyword stuffing, zero-width tricks, and abnormal density."""
    if not text:
        return GuardReport(
            is_suspicious=False,
            risk_level="low",
            keyword_density=0.0,
            detected_anomalies=[],
        )

    anomalies: list[str] = []

    # 1. Check for zero-width characters (common ATS injection trick)
    zero_width_pattern = re.compile(r"[\u200B-\u200D\uFEFF]")
    zw_matches = zero_width_pattern.findall(text)
    if len(zw_matches) > 3:
        anomalies.append(
            f"Detected {len(zw_matches)} hidden zero-width unicode characters (possible ATS injection)."
        )

    # 2. Skill density check (abnormal ratio of skills to total words)
    words = [w for w in re.split(r"\s+", text) if w.strip()]
    word_count = len(words)

    density = (total_skills_found / word_count) if word_count > 0 else 0.0

    # If more than 12% of entire resume words are raw technical keywords
    if word_count > 80 and density > 0.12:
        anomalies.append(
            f"Unusually high skill density ({density * 100:.1f}% of total words). Potential keyword stuffing."
        )

    # 3. Repeated token spam check (e.g. 'python python python python')
    token_repeats = re.findall(r"\b(\w+)(?:\s+\1){3,}\b", text.lower())
    if token_repeats:
        anomalies.append(
            f"Consecutive token repetition detected: '{', '.join(set(token_repeats))}'."
        )

    # 4. White text / hidden block indicator in raw text
    if re.search(r"(?i)(color:\s*(?:white|#fff|#ffffff)|font-size:\s*0\.?\d*px)", text):
        anomalies.append("CSS/styling contains zero-size font or white-on-white text pattern.")

    risk = "low"
    if len(anomalies) >= 2 or any("zero-width" in a for a in anomalies):
        risk = "high"
    elif len(anomalies) == 1:
        risk = "medium"

    return GuardReport(
        is_suspicious=len(anomalies) > 0,
        risk_level=risk,
        keyword_density=round(density, 4),
        detected_anomalies=anomalies,
    )
