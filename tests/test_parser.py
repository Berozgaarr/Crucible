from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

from crucible.core.parser import (
    classify_header,
    extract_experience_years,
    extract_name,
    extract_skills_with_evidence,
    parse_document,
)


def test_classify_header_standard_and_fuzzy():
    # Exact / standard
    assert classify_header("Skills") == "skills"
    assert classify_header("Work Experience") == "experience"
    assert classify_header("Education") == "education"
    assert classify_header("Projects") == "projects"

    # Fuzzy matches
    assert classify_header("Technical Proficiencies") == "skills"
    assert classify_header("What I've Built") == "projects"
    assert classify_header("Professional Background") == "experience"
    assert classify_header("Academic History") == "education"

    # Non-header text
    assert classify_header("Built distributed microservices with Python and Go.") is None


def test_symbol_safe_skill_extraction():
    text = (
        "Experienced engineer in C++, C#, and .NET frameworks. "
        "Also built web applications using Node.js and configured CI/CD pipelines. "
        "Proficient in Go and Python, but have not worked at Google."
    )
    vocab = {"C++", "C#", ".NET", "Node.js", "CI/CD", "Go", "Python", "Google", "Java"}

    matched, evidence = extract_skills_with_evidence(text, vocab)

    assert "C++" in matched
    assert "C#" in matched
    assert ".NET" in matched
    assert "Node.js" in matched
    assert "CI/CD" in matched
    assert "Go" in matched
    assert "Python" in matched
    # "Google" is in the text as company; should match if in vocab,
    # but Go shouldn't falsely trigger Google.
    assert "Java" not in matched

    # Evidence check
    assert "C++" in evidence
    assert "C#" in evidence
    assert "Experienced engineer in C++, C#, and .NET frameworks." in evidence["C++"]
    assert "Node.js" in evidence
    assert "Node.js" in evidence["Node.js"]


def test_experience_years_extraction():
    # Stated years
    phrase_1 = "Senior Software Engineer with 5+ years of experience in backend systems."
    assert extract_experience_years(phrase_1) == 5.0
    assert extract_experience_years("Over 3.5 yrs developing cloud platforms.") == 3.5
    assert extract_experience_years("10 years working across distributed teams.") == 10.0

    # Max years taken when multiple mentioned
    multi = "3 years in Python and 7+ years in distributed databases."
    assert extract_experience_years(multi) == 7.0

    # Unstated / ambiguous -> None (never default to 0.0)
    phrase_none = "Passionate software engineer building high performance apps."
    assert extract_experience_years(phrase_none) is None
    assert extract_experience_years("") is None


def test_name_heuristic_and_filename_fallback():
    # Clean name in first few lines
    clean_text = "Jane Doe\nSenior Backend Engineer\njane@example.com\n555-0199"
    assert extract_name(clean_text, "resume.pdf") == "Jane Doe"

    # Fallback to sanitized filename if lines look like email/phone/generic header
    fallback_text = "Curriculum Vitae\nsoftware engineer\ncontact: me@site.com\nphone: 123"
    assert extract_name(fallback_text, "John_Smith_Resume.pdf") == "John Smith"
    assert extract_name(fallback_text, "alice-walker-cv.pdf") == "alice walker"


def test_scanned_warning_and_parse_document_text(tmp_path: Path):
    # Short text < 50 chars triggers warning
    short_text = "Too short"
    doc_short = parse_document(
        raw_text=short_text, filename="scanned_resume.pdf", skill_vocab={"Python"}
    )
    assert doc_short.is_scanned_warning is True
    assert doc_short.name == "scanned"

    # Valid text document
    valid_text = (
        "Alice Johnson\n"
        "Lead Engineer with 6+ years of production experience.\n"
        "Proficiencies\n"
        "Developed scalable backends using Python and C++ with automated CI/CD.\n"
    )
    doc_valid = parse_document(
        raw_text=valid_text,
        filename="Alice_Johnson_CV.pdf",
        skill_vocab={"Python", "C++", "CI/CD", "Rust"},
    )
    assert doc_valid.is_scanned_warning is False
    assert doc_valid.name == "Alice Johnson"
    assert doc_valid.experience_years == 6.0
    assert doc_valid.skills == {"Python", "C++", "CI/CD"}
    assert "Python" in doc_valid.skill_evidence
    assert "C++" in doc_valid.skill_evidence


def test_education_level_extraction():
    from crucible.core.parser import extract_education_level

    assert extract_education_level("Candidate has a Ph.D. in Computer Science.") == "phd"
    assert extract_education_level("Earned a Master of Science in Data Analytics.") == "master"
    assert (
        extract_education_level("Bachelor of Science in Software Engineering, 2021.") == "bachelor"
    )
    assert extract_education_level("B.Tech in Information Technology.") == "bachelor"
    assert extract_education_level("Graduated with B.Eng in Electronic Engineering.") == "bachelor"
    assert extract_education_level("Holds M.Eng from Imperial College.") == "master"
    assert extract_education_level("Associate degree in Web Development.") == "associate"
    assert extract_education_level("Self-taught developer, no formal degree.") is None


def test_certifications_extraction():
    from crucible.core.parser import extract_certifications

    text = "Certified in AWS Solutions Architect and hold CKA credential."
    certs = extract_certifications(text)
    assert "AWS Certified" in certs
    assert "CKA" in certs
    assert "PMP" not in certs


def test_work_auth_extraction():
    from crucible.core.parser import extract_work_auth

    assert extract_work_auth("US Citizen authorized to work in United States.") == "citizen"
    assert extract_work_auth("Permanent Resident (Green Card holder).") == "permanent_resident"
    assert (
        extract_work_auth("Currently on F-1 OPT, will require visa sponsorship.")
        == "sponsorship_required"
    )


def test_parse_job_description():
    from crucible.core.parser import parse_job_description

    jd_text = (
        "Apex Systems looking for Senior Backend Developer. "
        "Requirements: 4+ years experience. Minimum Bachelor degree. "
        "Must know Python, Docker, PostgreSQL, and Git. "
        "AWS Certified preferred. US Citizen only."
    )
    parsed = parse_job_description(jd_text, title="Senior Backend")
    assert parsed["title"] == "Senior Backend"
    assert parsed["min_experience_years"] == 4.0
    assert parsed["required_education"] == "bachelor"
    assert "Python" in parsed["required_skills"]
    assert "AWS Certified" in parsed["required_certifications"]
    assert parsed["required_work_auth"] == "citizen"


if __name__ == "__main__":
    test_classify_header_standard_and_fuzzy()
    test_symbol_safe_skill_extraction()
    test_experience_years_extraction()
    test_name_heuristic_and_filename_fallback()
    test_scanned_warning_and_parse_document_text(Path("/tmp"))
    test_education_level_extraction()
    test_certifications_extraction()
    test_work_auth_extraction()
    test_parse_job_description()
    print("ALL TESTS PASSED")
