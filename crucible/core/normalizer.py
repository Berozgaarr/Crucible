"""Safe normalization routines and Spearman rank correlation validation."""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def normalize_semantic_scores(
    scores: Sequence[float],
    min_spread: float = 0.01,
) -> list[float]:
    """Safe bounded min-max normalization protecting against score clustering and zero-division."""
    if not scores:
        return []

    arr = np.asarray(scores, dtype=np.float32)
    s_min, s_max = float(np.min(arr)), float(np.max(arr))

    # Protect against zero division or collapsed spread (< min_spread)
    if (s_max - s_min) < min_spread or len(arr) <= 1:
        # Fallback linear stretch against typical MiniLM raw similarity boundaries [0.30, 0.85]
        normalized = np.clip((arr - 0.30) / 0.55, 0.0, 1.0)
    else:
        normalized = (arr - s_min) / (s_max - s_min)

    return [round(float(s), 4) for s in normalized]


def compute_experience_score(candidate_years: float | None, min_years: float | None) -> float:
    """Compute experience fit score with neutral midpoint for unknown experience."""
    if min_years is None or min_years <= 0:
        return 1.0

    if candidate_years is None:
        return 0.5  # Neutral midpoint for unstated/unknown tenure

    # Cap at 1.2 to reward seniority without overpowering skill matching
    return round(float(min(candidate_years / min_years, 1.2)), 4)


def compute_spearman_validation(
    manual_ranks: Sequence[float],
    system_ranks: Sequence[float],
) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient rho and p-value against human evaluation."""
    if len(manual_ranks) != len(system_ranks):
        raise ValueError("manual_ranks and system_ranks must have identical lengths")

    if len(manual_ranks) < 2:
        return 0.0, 1.0

    result = stats.spearmanr(manual_ranks, system_ranks)
    rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
    p_val = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])

    # Handle NaN gracefully if input ranks are constant
    return round(rho, 4), round(p_val, 4)


def compute_education_score(candidate_degree: str | None, required_degree: str | None) -> float:
    """Compute score based on education tier hierarchy."""
    from crucible.config import EDUCATION_LEVELS

    req_tier = EDUCATION_LEVELS.get((required_degree or "none").lower(), 0)
    if req_tier == 0:
        return 1.0

    cand_tier = EDUCATION_LEVELS.get((candidate_degree or "none").lower(), 0)
    if cand_tier >= req_tier:
        return 1.0
    # Scaled partial credit (e.g. Associate for Bachelor gives 2/3 = 0.67)
    return round(float(cand_tier / req_tier), 4)


def compute_certification_score(
    candidate_certs: set[str] | list[str], required_certs: list[str]
) -> float:
    """Compute overlap score for professional certifications."""
    if not required_certs:
        return 1.0
    cand_set = set(candidate_certs)
    req_set = set(required_certs)
    overlap = len(cand_set & req_set)
    return round(float(overlap / len(req_set)), 4)


def evaluate_knockouts(candidate: dict, jd: dict) -> list[str]:
    """Evaluate hard disqualification criteria. Returns list of failed knockout reasons."""
    from crucible.config import EDUCATION_LEVELS

    reasons: list[str] = []

    # 1. Experience knockout (if strict)
    min_exp = jd.get("min_experience_years")
    cand_exp = candidate.get("experience_years")
    if min_exp and cand_exp is not None and cand_exp < (min_exp * 0.5):
        reasons.append(f"Insufficient experience ({cand_exp}y vs required {min_exp}y)")

    # 2. Education knockout
    req_edu = jd.get("required_education")
    if req_edu and req_edu != "none":
        req_tier = EDUCATION_LEVELS.get(req_edu.lower(), 0)
        cand_edu = candidate.get("education_level")
        cand_tier = EDUCATION_LEVELS.get((cand_edu or "none").lower(), 0)
        if cand_tier < req_tier:
            reasons.append(
                f"Degree requirement not met ({cand_edu or 'None'} vs required {req_edu})"
            )

    # 3. Work Authorization knockout
    req_auth = jd.get("required_work_auth", "any")
    cand_auth = candidate.get("work_auth")
    if req_auth == "citizen" and cand_auth == "sponsorship_required":
        reasons.append("Requires visa sponsorship (Role requires US Citizenship)")

    # 4. Mandatory certifications knockout
    req_certs = jd.get("required_certifications", [])
    cand_certs = set(candidate.get("certifications", []))
    missing_certs = set(req_certs) - cand_certs
    if missing_certs:
        reasons.append(f"Missing mandatory certification(s): {', '.join(missing_certs)}")

    # 5. Location / Work Arrangement knockout (if non-remote role and candidate remote-only)
    target_loc = (jd.get("target_location") or "Remote").lower()
    cand_loc = (candidate.get("location") or "Remote").lower()
    if target_loc == "on-site" and cand_loc == "remote":
        reasons.append("Location conflict (Role is On-site, candidate requests Remote only)")

    return reasons


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> dict[str, float]:
    """Compute Reciprocal Rank Fusion (RRF) scores across multiple ranked candidate ID lists."""
    rrf_scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, item_id in enumerate(rank_list):
            if item_id not in rrf_scores:
                rrf_scores[item_id] = 0.0
            rrf_scores[item_id] += 1.0 / (k + (rank + 1))

    # Normalize max score to 1.0 if not empty
    if rrf_scores:
        max_val = max(rrf_scores.values())
        if max_val > 0:
            for item_id in rrf_scores:
                rrf_scores[item_id] = round(rrf_scores[item_id] / max_val, 4)

    return rrf_scores
