"""Resilient deterministic document parser for resumes and job descriptions."""

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    doc_id: str
    name: str
    raw_text: str
    skills: set[str]
    experience_years: float | None
    skill_evidence: dict[str, str] = field(default_factory=dict)
    education_level: str | None = None  # high_school, associate, bachelor, master, phd
    certifications: set[str] = field(default_factory=set)
    work_auth: str | None = None  # citizen, permanent_resident, sponsorship_required, any
    location: str | None = None
    is_scanned_warning: bool = False
    # Industry-standard JSON Resume & HR-XML fields
    email: str | None = None
    phone: str | None = None
    urls: dict[str, str] = field(default_factory=dict)  # github, linkedin, portfolio
    languages: dict[str, str] = field(default_factory=dict)
    security_clearance: str | None = None  # Secret, Top Secret, TS/SCI
    major: str | None = None
    gpa: float | None = None
    publications_count: int = 0
    patents_count: int = 0
    sections: dict[str, str] = field(default_factory=dict)


# Canonical section categories mapped to known phrases for fuzzy classification
SECTION_TAXONOMY: dict[str, list[str]] = {
    "skills": [
        "skills",
        "technical skills",
        "technical proficiencies",
        "proficiencies",
        "core competencies",
        "technologies",
        "tech stack",
    ],
    "experience": [
        "experience",
        "work experience",
        "employment history",
        "professional background",
        "career history",
        "work history",
        "professional experience",
    ],
    "projects": [
        "projects",
        "personal projects",
        "key projects",
        "what i've built",
        "selected projects",
        "academic projects",
    ],
    "education": [
        "education",
        "academic background",
        "academic history",
        "educational background",
        "qualifications",
    ],
}


def classify_header(line: str, cutoff: float = 0.75) -> str | None:
    """Fuzzy header classification using difflib with threshold cutoff."""
    cleaned = line.strip().lower().rstrip(":")
    if not cleaned or len(cleaned) > 50 or "\n" in cleaned:
        return None

    for category, phrases in SECTION_TAXONOMY.items():
        matches = difflib.get_close_matches(cleaned, phrases, n=1, cutoff=cutoff)
        if matches:
            return category
    return None


def extract_sections(text: str) -> dict[str, str]:
    """Segment document text into standard canonical sections."""
    if not text:
        return {}

    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current_sec = "summary"
    sections[current_sec] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        classified = classify_header(stripped)
        if classified:
            current_sec = classified
            if current_sec not in sections:
                sections[current_sec] = []
        else:
            sections[current_sec].append(stripped)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_skills_with_evidence(
    text: str, skill_vocabulary: set[str]
) -> tuple[set[str], dict[str, str]]:
    """Extract skills using symbol-safe regex and bind to verbatim context sentences."""
    if not text or not skill_vocabulary:
        return set(), {}

    # Split text into sentences for grounded evidence citations
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    matched_skills: set[str] = set()
    skill_evidence: dict[str, str] = {}

    for skill in skill_vocabulary:
        raw_skill = skill.strip()
        if not raw_skill:
            continue
        # Symbol-safe regex matching: handles C++, C#, .NET, Node.js, CI/CD
        # Allows trailing period at sentence end while preventing .word continuations
        # (e.g. .NET vs .NETwork)
        if raw_skill.endswith("."):
            pattern = rf"(?<![\w#+.]){re.escape(raw_skill)}(?![\w#+])"
        else:
            pattern = rf"(?<![\w#+.]){re.escape(raw_skill)}(?![\w#+])(?!\.[a-zA-Z0-9])"
        compiled = re.compile(pattern, re.IGNORECASE)

        for sentence in sentences:
            if compiled.search(sentence):
                matched_skills.add(raw_skill)
                # Keep the longest descriptive sentence as the strongest citation
                current = skill_evidence.get(raw_skill, "")
                if len(sentence) > len(current):
                    skill_evidence[raw_skill] = sentence

    return matched_skills, skill_evidence


def extract_experience_years(text: str) -> float | None:
    """Extract years of experience; return None if missing or unstated."""
    if not text:
        return None

    # Matches "5+ years", "3.5 yrs", "10 years"
    pattern = r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None

    years = [float(m) for m in matches]
    return max(years) if years else None


def extract_name(text: str, filename: str) -> str:
    """Extract candidate name using top lines heuristic with safe filename fallback."""
    stem = Path(filename).stem
    # Replace common trailing resume tokens from filename (e.g. Jane_Doe_Resume -> Jane Doe)
    fallback_name = re.sub(r"(?i)[-_]?(?:resume|cv|profile|summary)$", "", stem)
    fallback_name = re.sub(r"[-_]+", " ", fallback_name).strip()

    if not fallback_name:
        fallback_name = stem.replace("_", " ").replace("-", " ").strip()

    if len(text.strip()) < 50:
        return fallback_name

    lines = [line.strip() for line in text.splitlines() if line.strip()][:4]
    disallowed = re.compile(
        r"(?i)(resume|curriculum|vitae|email|phone|contact|engineer|developer|page|\d|@|https?://)"
    )

    for line in lines:
        words = line.split()
        if 2 <= len(words) <= 4 and not disallowed.search(line):
            if all(len(w) > 1 and w[0].isupper() for w in words):
                return line

    return fallback_name


def extract_education_level(text: str) -> str | None:
    """Deterministically extract highest education degree via regex."""
    if not text:
        return None

    # Order from highest to lowest degree
    phd_pat = r"\b(ph\.?d|d\.?phil|doctorate|doctor of philosophy)\b"
    master_pat = (
        r"\b(m\.?s|m\.?sc|master'?s|master of science|m\.?tech|m\.?eng|mba|"
        r"m\.?phil|m\.?a|magister|diplom)\b"
    )
    bachelor_pat = (
        r"\b(b\.?s|b\.?sc|bachelor'?s|bachelor of science|b\.?tech|b\.?e|"
        r"b\.?eng|b\.?a|undergraduate degree|licentiate)\b"
    )
    assoc_pat = r"\b(associate'?s|associate degree|hnd|foundation degree)\b"
    hs_pat = r"\b(high school|secondary education|ged|abitur|a-?levels?)\b"

    t = text.lower()
    if re.search(phd_pat, t):
        return "phd"
    if re.search(master_pat, t):
        return "master"
    if re.search(bachelor_pat, t):
        return "bachelor"
    if re.search(assoc_pat, t):
        return "associate"
    if re.search(hs_pat, t):
        return "high_school"

    return None


def extract_certifications(text: str, known_certs: dict[str, list[str]] | None = None) -> set[str]:
    """Extract professional credentials/certifications via alias match."""
    if not text:
        return set()

    from crucible.config import KNOWN_CERTIFICATIONS

    certs_to_check = known_certs or KNOWN_CERTIFICATIONS
    matched = set()
    t = text.lower()

    for canonical, aliases in certs_to_check.items():
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, t):
                matched.add(canonical)
                break

    return matched


def extract_work_auth(text: str) -> str | None:
    """Extract work authorization status from resume header or text."""
    if not text:
        return None
    t = text.lower()

    if re.search(r"\b(us citizen|u\.s\. citizen|citizen of the united states)\b", t):
        return "citizen"
    if re.search(r"\b(green card|permanent resident|us permanent resident)\b", t):
        return "permanent_resident"
    sponsor_pat = (
        r"\b(sponsorship required|require sponsorship|requires visa sponsorship|"
        r"need sponsorship|h-?1b)\b"
    )
    if re.search(sponsor_pat, t):
        return "sponsorship_required"
    if re.search(r"\b(cpt|opt|stem opt|f-?1 visa)\b", t):
        return "sponsorship_required"

    return None


def extract_location(text: str) -> str | None:
    """Extract location preference or city/remote indicator."""
    if not text:
        return None
    t = text.lower()
    if re.search(r"\b(remote|work from home|wfh|anywhere)\b", t):
        return "Remote"
    if re.search(r"\b(hybrid|flexible)\b", t):
        return "Hybrid"
    return "On-site"


def extract_contact_info(text: str) -> tuple[str | None, str | None, dict[str, str]]:
    """Extract email, phone, and profile URLs (GitHub, LinkedIn, Portfolio)."""
    if not text:
        return None, None, {}

    # Email extraction (RFC compliant regex)
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    email = email_match.group(0) if email_match else None

    # Phone extraction (standard formats: (123) 456-7890, +1-555-555-5555)
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    phone = phone_match.group(0) if phone_match else None

    # URLs
    urls: dict[str, str] = {}
    gh_match = re.search(r"\b(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)\b", text, re.I)
    if gh_match:
        urls["github"] = f"https://github.com/{gh_match.group(1)}"

    li_match = re.search(
        r"\b(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_-]+)\b", text, re.I
    )
    if li_match:
        urls["linkedin"] = f"https://linkedin.com/in/{li_match.group(1)}"

    return email, phone, urls


def extract_languages(text: str) -> dict[str, str]:
    """Extract spoken/written natural languages and fluency levels."""
    if not text:
        return {}
    known_langs = {
        "English": r"\b(english)\b",
        "Spanish": r"\b(spanish|español)\b",
        "French": r"\b(french|français)\b",
        "German": r"\b(german|deutsch)\b",
        "Mandarin": r"\b(mandarin|chinese)\b",
        "Hindi": r"\b(hindi)\b",
        "Japanese": r"\b(japanese)\b",
    }
    found = {}
    t_lower = text.lower()
    for lang, pat in known_langs.items():
        if re.search(pat, t_lower):
            found[lang] = "Fluent / Professional"
    return found


def extract_security_clearance(text: str) -> str | None:
    """Extract government / defense security clearance level."""
    if not text:
        return None
    t = text.lower()
    if re.search(r"\b(ts/sci|top secret/sci|top secret sci)\b", t):
        return "TS/SCI"
    if re.search(r"\b(top secret)\b", t):
        return "Top Secret"
    if re.search(r"\b(secret clearance|active secret)\b", t):
        return "Secret"
    if re.search(r"\b(public trust)\b", t):
        return "Public Trust"
    return None


def extract_academic_details(text: str) -> tuple[str | None, float | None]:
    """Extract college major and normalized GPA (on 4.0 scale)."""
    if not text:
        return None, None

    # Majors
    major_pat = re.compile(
        r"\b(?:in|of)\s+(computer science|data science|electrical engineering|"
        r"software engineering|information technology|mathematics|statistics|"
        r"artificial intelligence|physics)\b",
        re.I,
    )
    major_m = major_pat.search(text)
    major = major_m.group(1).title() if major_m else None

    # GPA: matches "3.85 GPA", "GPA: 3.9/4.0", "GPA of 3.7"
    gpa_pat = re.compile(r"\b(?:gpa[:\s]+)?([2-4]\.\d{1,2})(?:\s*/\s*4\.0)?\b", re.I)
    gpa_m = gpa_pat.search(text)
    gpa = float(gpa_m.group(1)) if gpa_m else None

    return major, gpa


def parse_job_description(text: str, title: str | None = None) -> dict:
    """Deterministically extract requirements from JD text/PDF."""
    from crucible.config import KNOWN_CERTIFICATIONS

    # Extract years
    exp_years = extract_experience_years(text) or 1.0

    # Extract degree
    degree = extract_education_level(text) or "bachelor"

    # Extract certifications
    certs = sorted(list(extract_certifications(text, KNOWN_CERTIFICATIONS)))

    # Work auth
    auth = (
        "citizen"
        if re.search(r"\b(us citizen only|citizen required|clearance required)\b", text.lower())
        else "any"
    )

    # Location
    loc = extract_location(text) or "Remote"

    # Extract skills using vocabulary from keyword aliases
    from crucible.core.keyword import CANONICAL_ALIASES

    all_skills_vocab = set(CANONICAL_ALIASES.values())
    skills_found, _ = extract_skills_with_evidence(text, all_skills_vocab)
    req_skills = sorted(list(skills_found))[:6]
    nice_skills = sorted(list(skills_found))[6:10]

    return {
        "id": "custom_uploaded_jd",
        "title": title or "Custom Job Description",
        "company": "Uploaded Organization",
        "min_experience_years": exp_years,
        "required_skills": req_skills or ["Python", "Git"],
        "nice_to_have_skills": nice_skills,
        "required_education": degree,
        "required_certifications": certs,
        "target_location": loc,
        "required_work_auth": auth,
    }


def extract_text_from_pdf(
    pdf_path: str | Path,
    enable_ocr_fallback: bool = True,
) -> tuple[str, bool]:
    """Extract text from PDF with optional OCR fallback for scanned documents."""
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages_text.append(t)

    full_text = "\n".join(pages_text).strip()
    is_scanned = len(full_text) < 50

    # OCR Fallback if text is empty/scanned and pytesseract is installed
    if is_scanned and enable_ocr_fallback:
        try:
            import pdf2image
            import pytesseract

            images = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=2)
            ocr_text_list = [pytesseract.image_to_string(img) for img in images]
            ocr_full = "\n".join(ocr_text_list).strip()
            if len(ocr_full) >= 50:
                full_text = ocr_full
                is_scanned = False
        except (ImportError, Exception):
            pass

    return full_text, is_scanned


def parse_document(
    file_path: str | Path | None = None,
    raw_text: str | None = None,
    filename: str | None = None,
    skill_vocab: set[str] | None = None,
) -> ParsedDocument:
    """Parse document from file or raw text into structured ParsedDocument."""
    is_scanned = False
    name_source = filename or "document"

    if raw_text is None:
        if file_path is None:
            raise ValueError("Either file_path or raw_text must be provided.")
        path_obj = Path(file_path)
        name_source = path_obj.name
        raw_text, is_scanned = extract_text_from_pdf(path_obj)
    else:
        is_scanned = len(raw_text.strip()) < 50

    doc_id = Path(name_source).stem
    doc_name = extract_name(raw_text, name_source)
    skills, evidence = extract_skills_with_evidence(raw_text, skill_vocab or set())
    years = extract_experience_years(raw_text)
    edu = extract_education_level(raw_text)
    certs = extract_certifications(raw_text)
    auth = extract_work_auth(raw_text)
    loc = extract_location(raw_text)

    # Industry standard ATS fields
    email, phone, urls = extract_contact_info(raw_text)
    langs = extract_languages(raw_text)
    clearance = extract_security_clearance(raw_text)
    major, gpa = extract_academic_details(raw_text)

    # Simple count heuristic for publications and patents
    pub_count = len(
        re.findall(r"\b(published|publication|ieee|acm|neurips|iclr|arxiv)\b", raw_text, re.I)
    )
    pat_count = len(re.findall(r"\b(patent|patented|uspto)\b", raw_text, re.I))
    sections = extract_sections(raw_text)

    return ParsedDocument(
        doc_id=doc_id,
        name=doc_name,
        raw_text=raw_text,
        skills=skills,
        experience_years=years,
        skill_evidence=evidence,
        education_level=edu,
        certifications=certs,
        work_auth=auth,
        location=loc,
        is_scanned_warning=is_scanned,
        email=email,
        phone=phone,
        urls=urls,
        languages=langs,
        security_clearance=clearance,
        major=major,
        gpa=gpa,
        publications_count=pub_count,
        patents_count=pat_count,
        sections=sections,
    )
