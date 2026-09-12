from crucible.core.bias import audit_job_description


def test_bias_audit_clean_jd():
    clean_jd = {
        "title": "Junior Full Stack Developer Intern",
        "min_experience_years": 1.0,
        "required_skills": ["Python", "React", "PostgreSQL", "Docker"],
        "required_education": "bachelor",
    }
    result = audit_job_description(clean_jd)
    assert result.inclusivity_score >= 80
    assert result.total_issues == 0


def test_bias_audit_detects_inflation_and_slang():
    biased_jd = {
        "title": "Junior Rockstar Developer Intern",
        "min_experience_years": 4.0,  # Inflation for junior
        "required_skills": [
            "Python",
            "FastAPI",
            "React",
            "Vue",
            "Angular",
            "Docker",
            "Kubernetes",
            "AWS",
            "GCP",
            "PostgreSQL",  # > 8 skills
        ],
        "required_education": "phd",  # Excessive for junior
    }
    raw_text = "We need a rockstar with a killer instinct who can work hard play hard."
    result = audit_job_description(biased_jd, raw_text=raw_text)

    assert result.total_issues >= 3
    assert result.inclusivity_score < 70
    categories = [f.category for f in result.findings]
    assert "language" in categories
    assert "credential_inflation" in categories
    assert "inflexible_stack" in categories
