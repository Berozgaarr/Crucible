from crucible.core.comparator import compare_candidates


def test_compare_candidates_score_and_skills_delta():
    cand_a = {
        "candidate_id": "c1",
        "name": "Sarah Chen",
        "rank": 1,
        "final_score": 0.92,
        "semantic_raw": 0.88,
        "keyword_score": 1.0,
        "experience_years": 3.0,
        "all_skills": ["Python", "FastAPI", "React", "PostgreSQL", "Docker", "Git"],
    }
    cand_b = {
        "candidate_id": "c2",
        "name": "Devon Miller",
        "rank": 3,
        "final_score": 0.74,
        "semantic_raw": 0.75,
        "keyword_score": 0.67,
        "experience_years": 1.0,
        "all_skills": ["Python", "FastAPI", "PostgreSQL", "Git"],
    }

    req_skills = ["Python", "FastAPI", "React", "PostgreSQL", "Docker", "Git"]
    res = compare_candidates(cand_a, cand_b, jd_required_skills=req_skills)

    assert res.candidate_a_id == "c1"
    assert res.candidate_b_id == "c2"
    assert res.score_delta > 0.15
    assert res.rank_a == 1
    assert res.rank_b == 3
    assert "React" in res.skills_a_only
    assert "Docker" in res.skills_a_only
    assert len(res.skills_b_only) == 0
    assert "Sarah Chen ranks #1 ahead of Devon Miller" in res.summary_verdict
    assert len(res.key_differentiators) > 0


def test_compare_candidates_close_tie():
    cand_a = {
        "candidate_id": "c1",
        "name": "Alice",
        "rank": 1,
        "final_score": 0.85,
        "semantic_raw": 0.80,
        "keyword_score": 0.80,
        "experience_years": 2.0,
        "skills": ["Python", "React"],
    }
    cand_b = {
        "candidate_id": "c2",
        "name": "Bob",
        "rank": 2,
        "final_score": 0.84,
        "semantic_raw": 0.79,
        "keyword_score": 0.80,
        "experience_years": 2.0,
        "skills": ["Python", "React"],
    }

    res = compare_candidates(cand_a, cand_b)
    assert abs(res.score_delta) <= 0.02
    assert "closely matched" in res.summary_verdict
