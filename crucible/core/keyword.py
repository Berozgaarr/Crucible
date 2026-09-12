"""Keyword coverage engine for transparent skill overlap scoring."""

from collections.abc import Iterable
from dataclasses import dataclass

# Canonical skill normalization mapping
CANONICAL_ALIASES: dict[str, str] = {
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "py": "Python",
    "python": "Python",
    "golang": "Go",
    "go": "Go",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    ".net": ".NET",
    "dotnet": ".NET",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "docker": "Docker",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "sql": "SQL",
    "redis": "Redis",
    "graphql": "GraphQL",
    "rest": "REST APIs",
    "rest api": "REST APIs",
    "restful": "REST APIs",
}


@dataclass
class KeywordMatchResult:
    keyword_score: float
    matched_required: list[str]
    missing_required: list[str]
    matched_nice_to_have: list[str]
    skill_depth_scores: dict[str, float] = None  # Skill depth breakdown (0.5 to 2.0)


def normalize_skill(skill: str) -> str:
    """Normalize skill string using canonical aliases or clean original casing."""
    trimmed = skill.strip()
    if not trimmed:
        return ""
    lookup = trimmed.lower()
    return CANONICAL_ALIASES.get(lookup, trimmed)


def normalize_skills(skills: Iterable[str]) -> set[str]:
    """Normalize an iterable of skills into a set of canonical skills."""
    return {normalize_skill(s) for s in skills if s and s.strip()}


def compute_skill_depth(skill: str, raw_text: str, sections: dict[str, str] | None = None) -> float:
    """Calculate contextual skill depth based on section placement, frequency, and action verbs."""
    if not raw_text or not skill:
        return 1.0

    import re

    # Base weight
    depth_multiplier = 1.0
    s_clean = re.escape(skill.lower())

    # 1. Section presence: experience & projects carry higher weight than standalone skills list
    if sections:
        exp_text = sections.get("experience", "").lower()
        proj_text = sections.get("projects", "").lower()
        skills_text = sections.get("skills", "").lower()

        in_exp = bool(re.search(rf"\b{s_clean}\b", exp_text))
        in_proj = bool(re.search(rf"\b{s_clean}\b", proj_text))
        in_list_only = bool(re.search(rf"\b{s_clean}\b", skills_text)) and not (in_exp or in_proj)

        if in_exp and in_proj:
            depth_multiplier += 0.5  # Practiced across both work and projects
        elif in_exp:
            depth_multiplier += 0.35  # Production work experience
        elif in_proj:
            depth_multiplier += 0.20  # Demonstrated in project work
        elif in_list_only:
            depth_multiplier -= 0.15  # Mere mention in a keyword bullet list

    # 2. Occurrence frequency bonus (capped)
    count = len(re.findall(rf"\b{s_clean}\b", raw_text.lower()))
    if count >= 3:
        depth_multiplier += 0.15
    elif count >= 2:
        depth_multiplier += 0.05

    return round(float(max(0.5, min(depth_multiplier, 1.8))), 3)


def compute_keyword_match(
    candidate_skills: Iterable[str],
    jd_required: Iterable[str],
    jd_nice_to_have: Iterable[str] | None = None,
    raw_text: str | None = None,
    sections: dict[str, str] | None = None,
) -> KeywordMatchResult:
    """Compute transparent keyword coverage score with contextual depth weighting."""
    cand_set = normalize_skills(candidate_skills)
    req_set = normalize_skills(jd_required)
    nice_set = normalize_skills(jd_nice_to_have or set())

    matched_req = sorted(cand_set & req_set)
    missing_req = sorted(req_set - cand_set)
    matched_nice = sorted(cand_set & nice_set)

    # Empty requirements guard to prevent division by zero
    if not req_set:
        return KeywordMatchResult(
            keyword_score=1.0,
            matched_required=[],
            missing_required=[],
            matched_nice_to_have=[],
            skill_depth_scores={},
        )

    # Compute depth weights for matched skills if raw_text is provided
    depth_scores: dict[str, float] = {}
    total_depth = 0.0
    for s in matched_req:
        d = compute_skill_depth(s, raw_text or "", sections) if raw_text else 1.0
        depth_scores[s] = d
        total_depth += d

    # Weighted coverage: base coverage modulated by average depth of matched skills
    avg_depth = (total_depth / len(matched_req)) if matched_req else 1.0
    base_coverage = len(matched_req) / len(req_set)
    weighted_coverage = base_coverage * min(avg_depth, 1.25)

    bonus = 0.0
    if nice_set:
        bonus = 0.1 * (len(matched_nice) / len(nice_set))

    score = round(min(weighted_coverage + bonus, 1.0), 4)

    return KeywordMatchResult(
        keyword_score=score,
        matched_required=matched_req,
        missing_required=missing_req,
        matched_nice_to_have=matched_nice,
        skill_depth_scores=depth_scores,
    )
