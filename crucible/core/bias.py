"""Deterministic Job Description Bias & Inclusivity Auditor (Zero-LLM).

Inspects job descriptions for exclusionary language, credential inflation,
and overly narrow technology locks without external LLM dependencies.
"""

import re
from dataclasses import dataclass, field


@dataclass
class BiasFinding:
    category: str  # "language", "credential_inflation", "inflexible_stack"
    severity: str  # "high", "medium", "low"
    title: str
    detected_phrase: str
    explanation: str
    suggested_alternative: str


@dataclass
class BiasAuditResult:
    total_issues: int
    inclusivity_score: int  # 0 to 100
    findings: list[BiasFinding] = field(default_factory=list)
    clean_summary: str = ""


# Deterministic pattern registry for non-inclusive / exclusionary phrasing
EXCLUSIONARY_PATTERNS = [
    {
        "regex": r"\b(ninja|rockstar|guru|wizard|superhero)\b",
        "category": "language",
        "severity": "medium",
        "title": "Hyper-competitive or Gender-skewed Slang",
        "explanation": "Terms like 'rockstar' or 'ninja' statistically discourage female and underrepresented candidates.",
        "suggestion": "Replace with 'skilled engineer', 'experienced developer', or 'specialist'.",
    },
    {
        "regex": r"\b(work\s+hard\s+play\s+hard|killer\s+instinct|aggressive)\b",
        "category": "language",
        "severity": "high",
        "title": "Burnout Culture & Aggressive Tone",
        "explanation": "Phrases like 'work hard play hard' signal burnout culture and discourage candidates with caretaking responsibilities.",
        "suggestion": "Replace with 'collaborative environment' or 'high-ownership culture'.",
    },
    {
        "regex": r"\b(native\s+english\s+speaker)\b",
        "category": "language",
        "severity": "high",
        "title": "Potential National Origin Bias",
        "explanation": "Mandating 'native' fluency can unfairly exclude qualified multilingual applicants.",
        "suggestion": "Replace with 'strong written and verbal English communication skills'.",
    },
    {
        "regex": r"\b(recent\s+grad(?:uate)?|digital\s+native|young\s+and\s+energetic)\b",
        "category": "language",
        "severity": "high",
        "title": "Age Discrimination Risk",
        "explanation": "Terms like 'digital native' or 'young and energetic' suggest age-based preference.",
        "suggestion": "Replace with 'proficient in modern web technologies' or 'eager to learn'.",
    },
]


def audit_job_description(jd_data: dict, raw_text: str | None = None) -> BiasAuditResult:
    """Run deterministic audit across JD title, requirements, and text."""
    findings: list[BiasFinding] = []

    title = jd_data.get("title", "")
    req_skills = jd_data.get("required_skills", [])
    min_exp = float(jd_data.get("min_experience_years") or 0.0)
    edu = (jd_data.get("required_education") or "").lower()

    text_corpus = f"{title} {raw_text or ''} {' '.join(req_skills)}"

    # 1. Regex Language Patterns
    for pat in EXCLUSIONARY_PATTERNS:
        matches = re.findall(pat["regex"], text_corpus, flags=re.IGNORECASE)
        if matches:
            phrase = matches[0] if isinstance(matches[0], str) else matches[0][0]
            findings.append(
                BiasFinding(
                    category=pat["category"],
                    severity=pat["severity"],
                    title=pat["title"],
                    detected_phrase=phrase,
                    explanation=pat["explanation"],
                    suggested_alternative=pat["suggestion"],
                )
            )

    # 2. Credential Inflation Checks
    is_junior_title = any(
        term in title.lower() for term in ["junior", "intern", "associate", "entry", "fresher"]
    )

    if is_junior_title and min_exp > 2.0:
        findings.append(
            BiasFinding(
                category="credential_inflation",
                severity="high",
                title="Experience Barrier on Entry-Level Role",
                detected_phrase=f"min_experience_years: {min_exp:.1f} years",
                explanation=f"A '{title}' role requiring {min_exp:.1f}+ years creates an artificial barrier for talented entry-level talent.",
                suggested_alternative="Lower minimum experience to 0–1 year or accept project portfolio demonstration.",
            )
        )

    if is_junior_title and edu in ["master", "phd"]:
        findings.append(
            BiasFinding(
                category="credential_inflation",
                severity="high",
                title="Excessive Education Prerequisite",
                detected_phrase=f"required_education: {edu.capitalize()}",
                explanation=f"Requiring a {edu.capitalize()} degree for a junior/intern opening filters out strong self-taught and bachelor candidates.",
                suggested_alternative="Change requirement to Bachelor's degree or equivalent practical experience.",
            )
        )

    # 3. Stack Hyper-Specialization / Tech Lock Check
    if len(req_skills) > 8:
        findings.append(
            BiasFinding(
                category="inflexible_stack",
                severity="medium",
                title="Excessive Mandatory Hard Skills Count",
                detected_phrase=f"{len(req_skills)} mandatory skills listed",
                explanation=f"Demanding {len(req_skills)} strictly required skills narrows candidate diversity. Research shows women apply only when meeting 100% of criteria.",
                suggested_alternative="Keep mandatory requirements to 4-5 core competencies and move adjacent tools to 'Nice to Have'.",
            )
        )

    # Score computation
    deductions = {"high": 25, "medium": 15, "low": 5}
    total_deduction = sum(deductions.get(f.severity, 10) for f in findings)
    inclusivity_score = max(0, 100 - total_deduction)

    if inclusivity_score >= 90:
        summary = "Highly inclusive JD with balanced criteria."
    elif inclusivity_score >= 70:
        summary = f"Good baseline with {len(findings)} minor inclusivity improvement areas."
    else:
        summary = f"Flagged {len(findings)} barriers risking exclusion of qualified candidates."

    return BiasAuditResult(
        total_issues=len(findings),
        inclusivity_score=inclusivity_score,
        findings=findings,
        clean_summary=summary,
    )
