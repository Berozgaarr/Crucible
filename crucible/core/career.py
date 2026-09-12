"""Career progression, title seniority indexing, and employment gap analysis."""

import re
from dataclasses import dataclass, field


@dataclass
class CareerAnalysis:
    velocity_score: float  # 0.0 - 1.0
    trajectory: str  # upward, steady, entry, unknown
    has_unexplained_gaps: bool
    detected_titles: list[str] = field(default_factory=list)
    gap_warnings: list[str] = field(default_factory=list)


# Standard title seniority rank ladder (1 to 5)
TITLE_LADDER: list[tuple[int, list[str]]] = [
    (1, ["intern", "trainee", "apprentice", "student", "co-op"]),
    (2, ["junior", "associate", "entry level", "entry-level", "graduate"]),
    (3, ["software engineer", "developer", "swe", "analyst", "consultant", "engineer"]),
    (4, ["senior", "sr.", "sr ", "lead", "specialist", "expert"]),
    (5, ["staff", "principal", "director", "head of", "vp", "architect", "fellow"]),
]


def score_title(title: str) -> int:
    """Assign seniority rank 1-5 to job title string."""
    t_lower = title.lower()
    for rank, keywords in reversed(TITLE_LADDER):
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", t_lower):
                return rank
    return 3  # Default mid-level if unclassified


def analyze_career_progression(text: str) -> CareerAnalysis:
    """Analyze career trajectory velocity and flag employment gaps."""
    if not text:
        return CareerAnalysis(
            velocity_score=0.5,
            trajectory="unknown",
            has_unexplained_gaps=False,
            detected_titles=[],
            gap_warnings=[],
        )

    # Extract common job title phrases
    title_pattern = re.compile(
        r"\b((?:senior|sr\.?|junior|lead|staff|principal|associate)?\s*"
        r"(?:software engineer|full[- ]?stack engineer|backend engineer|frontend engineer|"
        r"devops engineer|data engineer|cloud architect|system architect|developer))\b",
        re.IGNORECASE,
    )
    matches = title_pattern.findall(text)
    detected_titles = list(dict.fromkeys(matches))[:6]

    if not detected_titles:
        return CareerAnalysis(
            velocity_score=0.5,
            trajectory="steady",
            has_unexplained_gaps=False,
            detected_titles=[],
            gap_warnings=[],
        )

    scores = [score_title(t) for t in detected_titles]

    # Chronology heuristic: if last mentioned or first mentioned increases
    if len(scores) >= 2:
        # Check if rank progressed from lower to higher
        min_rank = min(scores)
        max_rank = max(scores)
        if max_rank > min_rank:
            trajectory = "upward"
            velocity = min(0.6 + (max_rank - min_rank) * 0.15, 1.0)
        else:
            trajectory = "steady"
            velocity = 0.65
    else:
        trajectory = "steady"
        velocity = 0.5 + (scores[0] * 0.08)

    # Check for long career gaps (> 2 years missing between dates)
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", text)]
    gap_warnings: list[str] = []
    has_gaps = False
    if len(years) >= 2:
        sorted_years = sorted(set(years))
        for i in range(len(sorted_years) - 1):
            diff = sorted_years[i + 1] - sorted_years[i]
            if diff > 2 and sorted_years[i + 1] <= 2026:
                has_gaps = True
                gap_warnings.append(
                    f"Career gap detected: {sorted_years[i]} to {sorted_years[i + 1]} ({diff} yrs)"
                )

    return CareerAnalysis(
        velocity_score=round(velocity, 4),
        trajectory=trajectory,
        has_unexplained_gaps=has_gaps,
        detected_titles=detected_titles,
        gap_warnings=gap_warnings[:3],
    )
