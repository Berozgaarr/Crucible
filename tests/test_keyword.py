"""Tests for crucible.core.keyword."""

from crucible.core.keyword import (
    compute_keyword_match,
    normalize_skill,
    normalize_skills,
)


def test_normalize_skill_and_aliases():
    assert normalize_skill("k8s") == "Kubernetes"
    assert normalize_skill("postgres") == "PostgreSQL"
    assert normalize_skill("postgresql") == "PostgreSQL"
    assert normalize_skill("js") == "JavaScript"
    assert normalize_skill("reactjs") == "React"
    assert normalize_skill("node") == "Node.js"
    assert normalize_skill("nodejs") == "Node.js"
    assert normalize_skill("ts") == "TypeScript"
    assert normalize_skill("  python  ") == "Python"
    # Unknown skill preserved with title casing or clean trimmed format
    assert normalize_skill("Solidity") == "Solidity"


def test_normalize_skills_set():
    raw = {"k8s", "postgres", "Docker", "js"}
    normalized = normalize_skills(raw)
    assert normalized == {"Kubernetes", "PostgreSQL", "Docker", "JavaScript"}


def test_keyword_match_perfect():
    candidate_skills = {"Python", "FastAPI", "Docker", "PostgreSQL"}
    jd_required = {"Python", "FastAPI", "Docker", "PostgreSQL"}

    result = compute_keyword_match(candidate_skills, jd_required)
    assert result.keyword_score == 1.0
    assert set(result.matched_required) == {"Python", "FastAPI", "Docker", "PostgreSQL"}
    assert result.missing_required == []


def test_keyword_match_zero():
    candidate_skills = {"Rust", "Solidity"}
    jd_required = {"Python", "FastAPI", "Docker"}

    result = compute_keyword_match(candidate_skills, jd_required)
    assert result.keyword_score == 0.0
    assert result.matched_required == []
    assert set(result.missing_required) == {"Python", "FastAPI", "Docker"}


def test_keyword_match_with_aliases():
    candidate_skills = {"k8s", "postgres", "reactjs", "nodejs"}
    jd_required = {"Kubernetes", "PostgreSQL", "React", "Node.js"}

    result = compute_keyword_match(candidate_skills, jd_required)
    assert result.keyword_score == 1.0
    assert set(result.matched_required) == {"Kubernetes", "PostgreSQL", "React", "Node.js"}
    assert result.missing_required == []


def test_keyword_match_partial_and_nice_to_have_bonus():
    candidate_skills = {"Python", "Docker", "AWS"}
    jd_required = {"Python", "Docker", "Kubernetes", "PostgreSQL"}  # 2 of 4 matched = 0.50
    jd_nice = {"AWS", "Terraform"}  # 1 of 2 matched = 0.5 * 0.1 bonus = 0.05

    result = compute_keyword_match(candidate_skills, jd_required, jd_nice)
    assert result.keyword_score == 0.55
    assert set(result.matched_required) == {"Python", "Docker"}
    assert set(result.missing_required) == {"Kubernetes", "PostgreSQL"}
    assert result.matched_nice_to_have == ["AWS"]


def test_keyword_match_empty_required_guard():
    candidate_skills = {"Python"}
    jd_required = set()
    jd_nice = {"AWS"}

    result = compute_keyword_match(candidate_skills, jd_required, jd_nice)
    assert result.keyword_score == 1.0
    assert result.matched_required == []
    assert result.missing_required == []
    assert result.matched_nice_to_have == []
