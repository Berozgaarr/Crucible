"""Unit tests for impact analysis, career velocity, guard, recency, and interview generation."""

from crucible.core.career import analyze_career_progression
from crucible.core.guard import inspect_resume_anomalies
from crucible.core.impact import analyze_impact
from crucible.core.interview import generate_interview_questions
from crucible.core.recency import analyze_skill_recency


def test_impact_analysis_metrics_and_verbs():
    text = (
        "Architected scalable microservices and spearheaded cloud migration. "
        "Reduced p99 latency by 45% and scaled throughput to 50k RPS, saving $120k annually."
    )
    result = analyze_impact(text)
    assert result.impact_score > 0.6
    assert result.metrics_count >= 3
    assert "architected" in result.detected_verbs or "spearheaded" in result.detected_verbs


def test_career_progression_trajectory():
    text = (
        "2018 - 2020: Junior Software Engineer at Startup A.\n"
        "2020 - 2023: Senior Software Engineer at TechCorp.\n"
        "2023 - Present: Staff Software Engineer at Enterprise Inc."
    )
    result = analyze_career_progression(text)
    assert result.trajectory == "upward"
    assert result.velocity_score >= 0.7


def test_guard_anti_cheat_anomaly_detection():
    # Normal resume
    clean_text = (
        "Software engineer with 4 years building web services using Python, Docker, and PostgreSQL."
    )
    clean_report = inspect_resume_anomalies(clean_text, total_skills_found=3)
    assert not clean_report.is_suspicious

    # Keyword stuffing anomaly
    spam_text = "python " * 50 + "docker kubernetes react"
    spam_report = inspect_resume_anomalies(spam_text, total_skills_found=20)
    assert spam_report.is_suspicious


def test_interview_question_generation():
    questions = generate_interview_questions(
        name="Alex Mercer",
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Kubernetes", "PostgreSQL"],
        evidence_quote="Scaled distributed Redis caching layer to handle 40k RPS.",
        detected_metrics=["40k RPS"],
    )
    assert len(questions) >= 3
    categories = {q.category for q in questions}
    assert "claim_verification" in categories
    assert "gap_probe" in categories


def test_skill_recency_decay():
    text = "2024 - Present: Working extensively with Python and FastAPI. 2015 - 2017: Built apps in Perl."
    recency = analyze_skill_recency(text, {"Python", "Perl"})
    assert recency["Python"].decay_weight == 1.0
    assert recency["Perl"].decay_weight == 0.40
