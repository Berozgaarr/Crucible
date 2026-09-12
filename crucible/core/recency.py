"""Temporal skill recency scoring and decay curve analysis."""

import re
from dataclasses import dataclass


@dataclass
class SkillRecency:
    skill: str
    last_seen_year: int | None
    decay_weight: float  # 1.0 (recent), 0.75 (mid), 0.40 (legacy)


CURRENT_YEAR = 2026


def extract_date_ranges(text: str) -> list[tuple[int, int]]:
    """Extract 4-digit year ranges from resume text (e.g. 2021 - Present, 2018 - 2022)."""
    # Matches patterns like '2019 - 2023', '2021 - Present', '2020 to Now'
    pattern = re.compile(
        r"\b(19\d\d|20\d\d)\s*(?:-|–|—|to)\s*(19\d\d|20\d\d|present|now|current)\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    ranges: list[tuple[int, int]] = []

    for start_str, end_str in matches:
        start_year = int(start_str)
        end_clean = end_str.lower()
        if end_clean in ("present", "now", "current"):
            end_year = CURRENT_YEAR
        else:
            end_year = int(end_str)
        ranges.append((start_year, end_year))

    return ranges


def calculate_recency_decay(last_year: int | None) -> float:
    """Compute temporal decay coefficient based on distance from current year."""
    if last_year is None:
        return 0.85  # Neutral default when year is unanchored

    years_ago = CURRENT_YEAR - last_year
    if years_ago <= 2:
        return 1.0  # Active / fresh skill
    elif years_ago <= 5:
        return 0.75  # Moderate recency
    else:
        return 0.40  # Legacy / stale skill


def analyze_skill_recency(
    text: str,
    skills: set[str],
) -> dict[str, SkillRecency]:
    """Map each skill to its most recent detected chronological anchor and decay weight."""
    results: dict[str, SkillRecency] = {}
    # Split by lines or sentence breaks to isolate chronological blocks
    blocks = [b.strip() for b in re.split(r"(?<=[.!?\n])\s+", text) if b.strip()]

    current_year_context: int | None = None

    for block in blocks:
        ranges = extract_date_ranges(block)
        if ranges:
            # Update active timeline context to latest end year found in block
            current_year_context = max(r[1] for r in ranges)

        for skill in skills:
            # Check if skill is mentioned in this line or nearby context
            if re.search(rf"\b{re.escape(skill)}\b", block, re.IGNORECASE):
                decay = calculate_recency_decay(current_year_context)
                if skill not in results or (
                    current_year_context
                    and (
                        results[skill].last_seen_year is None
                        or current_year_context > results[skill].last_seen_year
                    )
                ):
                    results[skill] = SkillRecency(
                        skill=skill,
                        last_seen_year=current_year_context,
                        decay_weight=decay,
                    )

    # Fill defaults for skills without specific date anchors
    for skill in skills:
        if skill not in results:
            results[skill] = SkillRecency(
                skill=skill,
                last_seen_year=None,
                decay_weight=0.85,
            )

    return results
