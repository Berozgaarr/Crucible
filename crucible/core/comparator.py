"""Deterministic Candidate Comparator Engine (Zero-LLM).

Computes head-to-head diffs between two candidates and generates an explainable,
grounded narrative explaining why Candidate A ranks above or below Candidate B.
"""

from dataclasses import dataclass, field


@dataclass
class CandidateComparison:
    candidate_a_id: str
    candidate_b_id: str
    candidate_a_name: str
    candidate_b_name: str
    score_delta: float  # a - b
    rank_a: int
    rank_b: int
    semantic_delta: float
    keyword_delta: float
    experience_delta: float
    skills_shared: list[str] = field(default_factory=list)
    skills_a_only: list[str] = field(default_factory=list)
    skills_b_only: list[str] = field(default_factory=list)
    summary_verdict: str = ""
    key_differentiators: list[str] = field(default_factory=list)


def compare_candidates(
    cand_a: dict,
    cand_b: dict,
    jd_required_skills: list[str] | None = None,
) -> CandidateComparison:
    """Perform deterministic head-to-head comparison between two candidates."""
    skills_a = set(
        cand_a.get("all_skills") or cand_a.get("skills") or cand_a.get("matched_skills") or []
    )
    skills_b = set(
        cand_b.get("all_skills") or cand_b.get("skills") or cand_b.get("matched_skills") or []
    )

    shared = sorted(list(skills_a & skills_b))
    a_only = sorted(list(skills_a - skills_b))
    b_only = sorted(list(skills_b - skills_a))

    score_a = float(cand_a.get("final_score", 0.0))
    score_b = float(cand_b.get("final_score", 0.0))
    score_delta = round(score_a - score_b, 4)

    sem_a = float(cand_a.get("semantic_raw", cand_a.get("semantic_norm", 0.0)))
    sem_b = float(cand_b.get("semantic_raw", cand_b.get("semantic_norm", 0.0)))
    sem_delta = round(sem_a - sem_b, 4)

    kw_a = float(cand_a.get("keyword_score", 0.0))
    kw_b = float(cand_b.get("keyword_score", 0.0))
    kw_delta = round(kw_a - kw_b, 4)

    exp_a = float(cand_a.get("experience_years") or 0.0)
    exp_b = float(cand_b.get("experience_years") or 0.0)
    exp_delta = round(exp_a - exp_b, 1)

    name_a = cand_a.get("name", "Candidate A")
    name_b = cand_b.get("name", "Candidate B")
    rank_a = int(cand_a.get("rank", 1))
    rank_b = int(cand_b.get("rank", 2))

    # Identify key differentiators
    differentiators = []
    if jd_required_skills:
        req_set = set(jd_required_skills)
        a_req_only = [s for s in a_only if s in req_set]
        b_req_only = [s for s in b_only if s in req_set]
        if a_req_only:
            differentiators.append(
                f"{name_a} holds required skills missing in {name_b}: {', '.join(a_req_only)}."
            )
        if b_req_only:
            differentiators.append(
                f"{name_b} holds required skills missing in {name_a}: {', '.join(b_req_only)}."
            )

    if abs(exp_delta) >= 1.0:
        lead = name_a if exp_delta > 0 else name_b
        lag = name_b if exp_delta > 0 else name_a
        differentiators.append(
            f"{lead} brings {abs(exp_delta):.1f} more years of verified experience than {lag}."
        )

    if abs(sem_delta) >= 0.08:
        lead = name_a if sem_delta > 0 else name_b
        lag = name_b if sem_delta > 0 else name_a
        differentiators.append(
            f"{lead} demonstrates significantly higher contextual semantic alignment with the job description."
        )

    if not differentiators:
        if score_delta > 0:
            differentiators.append(
                f"{name_a} holds a composite edge across blended ATS scorecard metrics."
            )
        elif score_delta < 0:
            differentiators.append(
                f"{name_b} holds a composite edge across blended ATS scorecard metrics."
            )
        else:
            differentiators.append("Both candidates exhibit nearly identical blended fit scores.")

    # Formulate verdict
    if score_delta > 0.02:
        verdict = (
            f"{name_a} ranks #{rank_a} ahead of {name_b} (#{rank_b}) with a +{score_delta * 100:.1f}% score advantage. "
            f"Primary factor: {differentiators[0]}"
        )
    elif score_delta < -0.02:
        verdict = (
            f"{name_b} ranks #{rank_b} ahead of {name_a} (#{rank_a}) by {abs(score_delta) * 100:.1f}%. "
            f"Primary factor: {differentiators[0]}"
        )
    else:
        verdict = f"{name_a} and {name_b} are closely matched with negligible score delta ({score_delta * 100:+.1f}%)."

    return CandidateComparison(
        candidate_a_id=cand_a.get("candidate_id", "cand_a"),
        candidate_b_id=cand_b.get("candidate_id", "cand_b"),
        candidate_a_name=name_a,
        candidate_b_name=name_b,
        score_delta=score_delta,
        rank_a=rank_a,
        rank_b=rank_b,
        semantic_delta=sem_delta,
        keyword_delta=kw_delta,
        experience_delta=exp_delta,
        skills_shared=shared,
        skills_a_only=a_only,
        skills_b_only=b_only,
        summary_verdict=verdict,
        key_differentiators=differentiators,
    )
