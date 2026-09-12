"""Composite candidate scorecard generation and rank ordering."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from crucible.core.normalizer import (
    compute_certification_score,
    compute_education_score,
    compute_experience_score,
    evaluate_knockouts,
    normalize_semantic_scores,
)


@dataclass
class CandidateScorecard:
    candidate_id: str
    name: str
    keyword_score: float
    semantic_raw: float
    semantic_norm: float
    experience_score: float
    education_score: float = 1.0
    cert_score: float = 1.0
    final_score: float = 0.0
    rank: int = 0
    knockout_flags: list[str] = field(default_factory=list)
    is_knocked_out: bool = False
    # Intelligence features
    impact_score: float = 0.5
    career_velocity: float = 0.5
    anti_cheat_clean: bool = True
    anti_cheat_flags: list[str] = field(default_factory=list)
    bm25_score: float = 0.0


def compute_scorecards(
    candidates: Sequence[dict],
    min_years: float | None = None,
    required_education: str | None = None,
    required_certifications: list[str] | None = None,
    jd_config: dict | None = None,
    w_kw: float = 0.35,
    w_sem: float = 0.40,
    w_exp: float = 0.15,
    w_edu: float = 0.05,
    w_cert: float = 0.05,
    w_bm25: float = 0.0,
    w_impact: float = 0.0,
    strict_knockouts: bool = False,
) -> list[CandidateScorecard]:
    """Calculate composite scores with ATS fields, normalize spread, and assign 1-based ranks."""
    if not candidates:
        return []

    raw_semantics = [float(c.get("semantic_raw", 0.0)) for c in candidates]
    norm_semantics = normalize_semantic_scores(raw_semantics)

    scorecards: list[CandidateScorecard] = []
    for i, c in enumerate(candidates):
        cid = str(c.get("candidate_id", f"cand_{i}"))
        cname = str(c.get("name", "Unknown Candidate"))
        kw_score = float(c.get("keyword_score", 0.0))
        sem_raw = raw_semantics[i]
        sem_norm = norm_semantics[i]
        cand_exp = c.get("experience_years")
        exp_score = compute_experience_score(cand_exp, min_years)

        # ATS fields
        cand_edu = c.get("education_level")
        cand_certs = c.get("certifications", set())
        edu_score = compute_education_score(cand_edu, required_education)
        cert_score = compute_certification_score(cand_certs, required_certifications or [])

        # Dynamic intelligence signals
        cand_impact = float(c.get("impact_score", 0.5))
        cand_bm25 = float(c.get("bm25_score", 0.0))

        # Knockouts
        knockouts: list[str] = []
        if jd_config:
            knockouts = evaluate_knockouts(c, jd_config)
        is_ko = len(knockouts) > 0

        # Composite blend formula: balances core rubric with intelligent signals
        composite = (
            (w_kw * kw_score)
            + (w_sem * sem_norm)
            + (w_exp * exp_score)
            + (w_edu * edu_score)
            + (w_cert * cert_score)
            + (w_bm25 * cand_bm25)
            + (w_impact * cand_impact)
        )

        # If strict knockout, demote score to bottom tier
        if strict_knockouts and is_ko:
            composite = composite * 0.1

        final_score = round(float(composite), 4)

        scorecards.append(
            CandidateScorecard(
                candidate_id=cid,
                name=cname,
                keyword_score=kw_score,
                semantic_raw=sem_raw,
                semantic_norm=sem_norm,
                experience_score=exp_score,
                education_score=edu_score,
                cert_score=cert_score,
                final_score=final_score,
                rank=0,  # Assigned after sorting
                knockout_flags=knockouts,
                is_knocked_out=is_ko,
                impact_score=float(c.get("impact_score", 0.5)),
                career_velocity=float(c.get("career_velocity", 0.5)),
                anti_cheat_clean=bool(c.get("anti_cheat_clean", True)),
                anti_cheat_flags=list(c.get("anti_cheat_flags", [])),
                bm25_score=float(c.get("bm25_score", 0.0)),
            )
        )

    # Sort descending by final_score, breaking ties by semantic_norm then keyword_score
    scorecards.sort(
        key=lambda s: (s.final_score, s.semantic_norm, s.keyword_score),
        reverse=True,
    )

    # Assign 1-indexed ranks
    for rank_idx, scorecard in enumerate(scorecards, start=1):
        scorecard.rank = rank_idx

    return scorecards
