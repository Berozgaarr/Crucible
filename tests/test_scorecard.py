"""Tests for crucible.core.normalizer and crucible.core.scorecard."""

import pytest

from crucible.core.normalizer import (
    compute_experience_score,
    compute_spearman_validation,
    normalize_semantic_scores,
)
from crucible.core.scorecard import CandidateScorecard, compute_scorecards


def test_normalize_semantic_scores_spread():
    # Tight clustered raw scores
    raw = [0.78, 0.80, 0.82]
    norm = normalize_semantic_scores(raw)
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0


def test_normalize_semantic_scores_fallback():
    # Identical values or delta < 0.05
    raw = [0.75, 0.76]
    norm = normalize_semantic_scores(raw)
    # Stretch against [0.30, 0.85]: (0.75 - 0.30)/0.55 = 0.8182
    assert 0.80 < norm[0] < 0.83
    assert 0.81 < norm[1] < 0.85

    # Single element
    assert normalize_semantic_scores([0.65]) == [round((0.65 - 0.30) / 0.55, 4)]
    # Empty
    assert normalize_semantic_scores([]) == []


def test_compute_experience_score():
    # Unknown experience returns 0.5 neutral midpoint
    assert compute_experience_score(None, 5.0) == 0.5

    # Min years unstated or 0 returns 1.0
    assert compute_experience_score(3.0, None) == 1.0
    assert compute_experience_score(3.0, 0.0) == 1.0

    # Overqualified years capped at 1.2
    assert compute_experience_score(10.0, 5.0) == 1.2

    # Underqualified years
    assert compute_experience_score(2.5, 5.0) == 0.5
    assert compute_experience_score(4.0, 5.0) == 0.8


def test_compute_scorecards_ranking():
    candidates = [
        {
            "candidate_id": "c1",
            "name": "Alice Backend",
            "keyword_score": 0.90,
            "semantic_raw": 0.82,
            "experience_years": 5.0,
        },
        {
            "candidate_id": "c2",
            "name": "Bob Frontend",
            "keyword_score": 0.40,
            "semantic_raw": 0.55,
            "experience_years": 2.0,
        },
        {
            "candidate_id": "c3",
            "name": "Carol Junior",
            "keyword_score": 0.70,
            "semantic_raw": 0.72,
            "experience_years": None,
        },
    ]

    cards = compute_scorecards(candidates, min_years=4.0)
    assert len(cards) == 3

    # Must be sorted descending by final score with 1-based ranks
    assert cards[0].rank == 1
    assert cards[1].rank == 2
    assert cards[2].rank == 3

    assert cards[0].candidate_id == "c1"
    assert cards[0].final_score > cards[1].final_score > cards[2].final_score
    assert isinstance(cards[0], CandidateScorecard)


def test_spearman_validation():
    # Perfect concordant ranking
    manual = [3.0, 3.0, 2.0, 2.0, 1.0]
    system = [5.0, 4.0, 3.0, 2.0, 1.0]
    rho, p_val = compute_spearman_validation(manual, system)

    assert rho > 0.90
    assert p_val < 0.05

    # Inverse ranking
    rho_inv, _ = compute_spearman_validation([1, 2, 3], [3, 2, 1])
    assert rho_inv == -1.0

    # Mismatched length raises error
    with pytest.raises(ValueError):
        compute_spearman_validation([1, 2], [1])


def test_ats_education_and_cert_scoring():
    from crucible.core.normalizer import (
        compute_certification_score,
        compute_education_score,
    )

    # Education scoring
    assert compute_education_score("phd", "bachelor") == 1.0
    assert compute_education_score("bachelor", "bachelor") == 1.0
    assert compute_education_score("associate", "bachelor") < 1.0
    assert compute_education_score(None, "bachelor") == 0.0

    # Cert scoring
    assert compute_certification_score({"AWS Certified", "CKA"}, ["AWS Certified"]) == 1.0
    assert compute_certification_score({"PMP"}, ["AWS Certified", "CKA"]) == 0.0
    assert compute_certification_score({"AWS Certified"}, ["AWS Certified", "CKA"]) == 0.5


def test_knockouts_evaluation():
    from crucible.core.normalizer import evaluate_knockouts

    jd = {
        "min_experience_years": 4.0,
        "required_education": "bachelor",
        "required_work_auth": "citizen",
        "required_certifications": ["AWS Certified"],
    }

    # Qualifying candidate
    cand_good = {
        "experience_years": 4.0,
        "education_level": "bachelor",
        "work_auth": "citizen",
        "certifications": {"AWS Certified"},
    }
    assert evaluate_knockouts(cand_good, jd) == []

    # Disqualified on work auth, education, and location
    cand_bad = {
        "experience_years": 0.5,
        "education_level": "high_school",
        "work_auth": "sponsorship_required",
        "location": "Remote",
        "certifications": set(),
    }
    jd_onsite = dict(jd, target_location="On-site")
    reasons = evaluate_knockouts(cand_bad, jd_onsite)
    assert len(reasons) >= 4
    assert any("experience" in r.lower() for r in reasons)
    assert any("degree" in r.lower() for r in reasons)
    assert any("sponsorship" in r.lower() for r in reasons)
    assert any("location" in r.lower() for r in reasons)
