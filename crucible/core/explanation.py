"""Grounded narrative explanation generator citing verbatim resume quote evidence."""

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class ExplanationResult:
    candidate_id: str
    narrative: str
    matched_skills: list[str]
    missing_skills: list[str]
    cited_quote: str | None


def format_cited_quote(quote: str | None, max_len: int = 160) -> str | None:
    """Clean and safely truncate verbatim quote to max_len with ellipsis."""
    if not quote:
        return None

    # Collapse excessive whitespace and internal newlines
    cleaned = re.sub(r"\s+", " ", quote).strip().strip("\"'")
    if not cleaned:
        return None

    if len(cleaned) <= max_len:
        return cleaned

    # Truncate at word boundary to avoid awkward word cuts
    truncated = cleaned[: max_len - 3].rsplit(" ", 1)[0]
    return f"{truncated}..."


def generate_narrative_explanation(
    name: str,
    matched_required: Sequence[str],
    missing_required: Sequence[str],
    top_evidence_quote: str | None = None,
    education_level: str | None = None,
    certifications: Sequence[str] | None = None,
    knockout_flags: Sequence[str] | None = None,
) -> str:
    """Synthesize deterministic evaluative verdict citing verbatim resume quote and ATS criteria."""
    cand_name = name.strip() or "Candidate"
    matched = list(matched_required)
    missing = list(missing_required)
    cleaned_quote = format_cited_quote(top_evidence_quote)

    if matched:
        top_skills = ", ".join(matched[:3])
        strength = f"{cand_name} is a strong match on {top_skills}"
        if cleaned_quote:
            strength += f', demonstrated through: "{cleaned_quote}".'
        else:
            strength += "."
    else:
        strength = f"{cand_name} does not match any primary required skills."

    if missing:
        gap = f" The primary gap is {', '.join(missing)}."
    else:
        gap = " No required competencies are missing."

    extra_clauses = []
    if education_level:
        extra_clauses.append(f"Holds {education_level.replace('_', ' ').title()} qualification")
    if certifications:
        extra_clauses.append(f"Certified in {', '.join(certifications)}")

    credentials = f" {' and '.join(extra_clauses)}." if extra_clauses else ""

    ko_clause = ""
    if knockout_flags:
        ko_clause = f" Warning: Failed knockout criteria ({'; '.join(knockout_flags)})."

    return strength + gap + credentials + ko_clause


def explain_candidate(
    candidate_id: str,
    name: str,
    matched_required: Sequence[str],
    missing_required: Sequence[str],
    top_evidence_quote: str | None = None,
    education_level: str | None = None,
    certifications: Sequence[str] | None = None,
    knockout_flags: Sequence[str] | None = None,
) -> ExplanationResult:
    """Build structured ExplanationResult object for downstream UI and reporting."""
    narrative = generate_narrative_explanation(
        name=name,
        matched_required=matched_required,
        missing_required=missing_required,
        top_evidence_quote=top_evidence_quote,
        education_level=education_level,
        certifications=certifications,
        knockout_flags=knockout_flags,
    )
    return ExplanationResult(
        candidate_id=candidate_id,
        narrative=narrative,
        matched_skills=list(matched_required),
        missing_skills=list(missing_required),
        cited_quote=format_cited_quote(top_evidence_quote),
    )
