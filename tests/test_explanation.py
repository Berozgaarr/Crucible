"""Tests for crucible.core.explanation."""

from crucible.core.explanation import (
    ExplanationResult,
    explain_candidate,
    format_cited_quote,
    generate_narrative_explanation,
)


def test_full_match_positive_verdict_no_gap():
    name = "Alex Mercer"
    matched = ["Python", "FastAPI", "PostgreSQL", "Docker"]
    missing = []
    quote = "Designed and deployed high throughput FastAPI microservices with PostgreSQL."

    narrative = generate_narrative_explanation(name, matched, missing, quote)
    assert "Alex Mercer is a strong match on Python, FastAPI, PostgreSQL" in narrative
    assert f'demonstrated through: "{quote}"' in narrative
    assert "No required competencies are missing." in narrative
    assert "primary gap" not in narrative


def test_partial_match_with_gap_clause():
    name = "Samira Khan"
    matched = ["React", "TypeScript"]
    missing = ["Kubernetes", "AWS"]
    quote = "Led frontend architecture refactoring using React and TypeScript."

    narrative = generate_narrative_explanation(name, matched, missing, quote)
    assert "Samira Khan is a strong match on React, TypeScript" in narrative
    assert "The primary gap is Kubernetes, AWS." in narrative


def test_zero_match_candidate():
    name = "Jordan Lee"
    matched = []
    missing = ["Python", "Docker"]

    narrative = generate_narrative_explanation(name, matched, missing, None)
    assert "Jordan Lee does not match any primary required skills." in narrative
    assert "The primary gap is Python, Docker." in narrative


def test_verbatim_quote_preservation_and_truncation():
    short_quote = "Developed scalable REST API with C++ and CI/CD."
    assert format_cited_quote(short_quote) == short_quote

    # Long quote over 160 characters
    long_quote = (
        "Architected enterprise distributed cloud microservices infrastructure utilizing "
        "Python, Docker containers, Kubernetes clusters, and asynchronous PostgreSQL pools "
        "across multiple AWS availability zones."
    )

    formatted = format_cited_quote(long_quote, max_len=120)
    assert len(formatted) <= 120
    assert formatted.endswith("...")
    assert "Architected enterprise distributed cloud" in formatted


def test_empty_and_edge_inputs():
    # Empty strings, None values, whitespace
    assert format_cited_quote(None) is None
    assert format_cited_quote("   ") is None

    result = explain_candidate(
        candidate_id="cand_99",
        name="",
        matched_required=[],
        missing_required=[],
        top_evidence_quote=None,
    )
    assert isinstance(result, ExplanationResult)
    assert result.candidate_id == "cand_99"
    assert "Candidate does not match any primary required skills." in result.narrative
    assert "No required competencies are missing." in result.narrative
    assert result.cited_quote is None
