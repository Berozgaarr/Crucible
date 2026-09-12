import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from crucible.config import DEFAULT_JD, DEMO_CANDIDATES, PRESET_JDS
from crucible.core.bias import audit_job_description
from crucible.core.bm25 import BM25Index
from crucible.core.career import analyze_career_progression
from crucible.core.comparator import compare_candidates
from crucible.core.explanation import explain_candidate
from crucible.core.guard import inspect_resume_anomalies
from crucible.core.impact import analyze_impact
from crucible.core.interview import generate_interview_questions
from crucible.core.keyword import compute_keyword_match
from crucible.core.parser import extract_sections, parse_document, parse_job_description
from crucible.core.scorecard import compute_scorecards
from crucible.core.semantic import SemanticEngine

app = FastAPI(title="Crucible", description="Deterministic Zero-LLM Shortlisting Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mutable in-memory active job description
ACTIVE_JD: dict = dict(DEFAULT_JD)


def get_alternative_role(candidate_skills: list[str]) -> str | None:
    """Recommend best alternative role for lower-ranked candidates based on skill match."""
    cand_set = set(candidate_skills)
    best_role = None
    best_overlap = 0
    from crucible.config import ALTERNATIVE_ROLES

    for role, skills in ALTERNATIVE_ROLES.items():
        overlap = len(cand_set & set(skills))
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = role
    return best_role if best_overlap > 0 else None


# In-memory semantic computation cache keyed by (jd_id, candidate_count)
_SEMANTIC_CACHE: dict[tuple, dict] = {}


def evaluate_candidates(
    mode: str = "hybrid",
    w_kw: float = 35.0,
    w_sem: float = 40.0,
    w_exp: float = 15.0,
    w_edu: float = 5.0,
    w_cert: float = 5.0,
    name_blind: bool = False,
    strict_knockouts: bool = False,
) -> list[dict]:
    """Calculate scores across candidates according to ranking mode, weights, and active JD."""
    req_skills = ACTIVE_JD.get("required_skills", [])
    nice_skills = ACTIVE_JD.get("nice_to_have_skills", [])
    min_exp = ACTIVE_JD.get("min_experience_years", 1.0)
    req_edu = ACTIVE_JD.get("required_education")
    req_certs = ACTIVE_JD.get("required_certifications", [])

    # Normalize slider weights
    total_w = (w_kw + w_sem + w_exp + w_edu + w_cert) or 100.0
    weight_kw = w_kw / total_w
    weight_sem = w_sem / total_w
    weight_exp = w_exp / total_w
    weight_edu = w_edu / total_w
    weight_cert = w_cert / total_w

    if mode == "keyword":
        weight_kw, weight_sem, weight_exp, weight_edu, weight_cert = 1.0, 0.0, 0.0, 0.0, 0.0
    elif mode == "semantic":
        weight_kw, weight_sem, weight_exp, weight_edu, weight_cert = 0.0, 1.0, 0.0, 0.0, 0.0

    # Pre-index candidates for BM25 Okapi lexical scoring
    corpus = [c.get("raw_text", "") for c in DEMO_CANDIDATES]
    bm25 = BM25Index().fit(corpus)
    jd_query = f"{ACTIVE_JD.get('title', '')} {' '.join(req_skills)} {' '.join(nice_skills)}"
    bm25_scores = bm25.score(jd_query)
    max_bm25 = max(bm25_scores) if (bm25_scores and max(bm25_scores) > 0) else 1.0

    # Cache key for expensive neural semantic calculations
    jd_id = ACTIVE_JD.get("id", "custom")
    cache_key = (jd_id, len(DEMO_CANDIDATES))

    if cache_key not in _SEMANTIC_CACHE:
        sem_engine = SemanticEngine.get_instance()
        jd_full_prompt = (
            f"{ACTIVE_JD.get('title', '')} at {ACTIVE_JD.get('company', '')}. "
            f"Requirements: {', '.join(req_skills)}. "
            f"Preferred: {', '.join(nice_skills)}."
        )

        candidate_sections_list = [extract_sections(c.get("raw_text", "")) for c in DEMO_CANDIDATES]

        hierarchical_semantic_scores = sem_engine.compute_hierarchical_similarity(
            candidate_sections_list=candidate_sections_list,
            candidate_full_texts=corpus,
            jd_text=jd_full_prompt,
        )

        # First pass prelim ranking for top-8 cross-encoder re-ranking
        prelim_tuples = []
        for idx, c in enumerate(DEMO_CANDIDATES):
            s_raw = (
                hierarchical_semantic_scores[idx]
                if idx < len(hierarchical_semantic_scores)
                else 0.5
            )
            kw_dummy = compute_keyword_match(
                c.get("skills", []), req_skills, nice_skills
            ).keyword_score
            bm_norm = round(bm25_scores[idx] / max_bm25, 4) if idx < len(bm25_scores) else 0.0
            p_score = 0.45 * s_raw + 0.35 * kw_dummy + 0.20 * bm_norm
            prelim_tuples.append((idx, p_score, s_raw))

        prelim_sorted = sorted(prelim_tuples, key=lambda x: x[1], reverse=True)
        top_k_indices = [x[0] for x in prelim_sorted[:8]]
        top_k_texts = [corpus[i] for i in top_k_indices]

        cross_scores = sem_engine.rerank_top_k(jd_full_prompt, top_k_texts, top_k=8)
        calibrated_map = {}
        for rank_pos, c_idx in enumerate(top_k_indices):
            if rank_pos < len(cross_scores):
                s_raw = hierarchical_semantic_scores[c_idx]
                calibrated_map[c_idx] = round(0.70 * cross_scores[rank_pos] + 0.30 * s_raw, 4)

        final_semantics = {}
        for idx in range(len(DEMO_CANDIDATES)):
            final_semantics[idx] = calibrated_map.get(idx, hierarchical_semantic_scores[idx])

        _SEMANTIC_CACHE[cache_key] = {
            "sections": candidate_sections_list,
            "semantics": final_semantics,
        }

    cached_data = _SEMANTIC_CACHE[cache_key]
    candidate_sections_list = cached_data["sections"]
    cached_semantics = cached_data["semantics"]

    # Step 2: Contextual Keyword Depth & Deep Intelligence Extraction
    cand_records: list[dict] = []
    for idx, c in enumerate(DEMO_CANDIDATES):
        raw_text = c.get("raw_text", "")
        cand_sections = candidate_sections_list[idx]

        kw_res = compute_keyword_match(
            c.get("skills", []),
            req_skills,
            nice_skills,
            raw_text=raw_text,
            sections=cand_sections,
        )
        display_name = f"Candidate #{c['candidate_id'].split('_')[-1]}" if name_blind else c["name"]

        impact = analyze_impact(raw_text)
        career = analyze_career_progression(raw_text)
        guard = inspect_resume_anomalies(raw_text, len(c.get("skills", [])))
        bm25_norm = round(bm25_scores[idx] / max_bm25, 4) if idx < len(bm25_scores) else 0.0
        sem_score = cached_semantics.get(idx, c.get("semantic_raw", 0.5))

        cand_records.append(
            {
                "candidate_id": c["candidate_id"],
                "name": display_name,
                "keyword_score": kw_res.keyword_score,
                "semantic_raw": sem_score,
                "experience_years": c.get("experience_years"),
                "education_level": c.get("education_level", "bachelor"),
                "certifications": set(c.get("certifications", set())),
                "work_auth": c.get("work_auth", "citizen"),
                "location": c.get("location", "Remote"),
                "matched_skills": kw_res.matched_required,
                "missing_skills": kw_res.missing_required,
                "all_skills": c.get("skills", []),
                "evidence": c.get("evidence"),
                "impact_score": impact.impact_score,
                "detected_metrics": impact.detected_metrics,
                "career_velocity": career.velocity_score,
                "career_trajectory": career.trajectory,
                "career_gaps": career.gap_warnings,
                "anti_cheat_clean": not guard.is_suspicious,
                "anti_cheat_flags": guard.detected_anomalies,
                "bm25_score": bm25_norm,
                "raw_text": raw_text,
            }
        )

    # Step 3: Compute normalized composite scorecards with hybrid BM25 + Impact blend
    scorecards = compute_scorecards(
        cand_records,
        min_years=min_exp,
        required_education=req_edu,
        required_certifications=req_certs,
        jd_config=ACTIVE_JD,
        w_kw=weight_kw,
        w_sem=weight_sem,
        w_exp=weight_exp,
        w_edu=weight_edu,
        w_cert=weight_cert,
        w_bm25=0.10 if mode == "hybrid" else 0.0,
        w_impact=0.05 if mode == "hybrid" else 0.0,
        strict_knockouts=strict_knockouts,
    )

    # Step 3: Attach explanations, interview questions & cross-role routing
    enriched: list[dict] = []

    for sc in scorecards:
        rec = next(r for r in cand_records if r["candidate_id"] == sc.candidate_id)
        expl = explain_candidate(
            candidate_id=sc.candidate_id,
            name=sc.name,
            matched_required=rec["matched_skills"],
            missing_required=rec["missing_skills"],
            top_evidence_quote=rec.get("evidence"),
            education_level=rec.get("education_level"),
            certifications=list(rec.get("certifications", [])),
            knockout_flags=sc.knockout_flags,
        )
        alt_role = get_alternative_role(rec.get("all_skills", []))

        # Synthesize technical interview questions
        questions = generate_interview_questions(
            name=sc.name,
            matched_skills=rec["matched_skills"],
            missing_skills=rec["missing_skills"],
            evidence_quote=rec.get("evidence"),
            detected_metrics=rec.get("detected_metrics"),
        )

        enriched.append(
            {
                "candidate_id": sc.candidate_id,
                "name": sc.name,
                "rank": sc.rank,
                "keyword_score": sc.keyword_score,
                "semantic_raw": sc.semantic_raw,
                "semantic_norm": sc.semantic_norm,
                "experience_score": sc.experience_score,
                "education_score": sc.education_score,
                "cert_score": sc.cert_score,
                "final_score": sc.final_score,
                "experience_years": rec.get("experience_years"),
                "education_level": rec.get("education_level"),
                "certifications": list(rec.get("certifications", [])),
                "matched_skills": rec["matched_skills"],
                "missing_skills": rec["missing_skills"],
                "cited_quote": expl.cited_quote,
                "narrative": expl.narrative,
                "recommended_role": alt_role
                if (sc.final_score < 0.55 or sc.keyword_score == 0)
                else None,
                "knockout_flags": sc.knockout_flags,
                "is_knocked_out": sc.is_knocked_out,
                # New ATS & intelligence properties
                "impact_score": sc.impact_score,
                "career_velocity": sc.career_velocity,
                "career_trajectory": rec.get("career_trajectory"),
                "career_gaps": rec.get("career_gaps"),
                "anti_cheat_clean": sc.anti_cheat_clean,
                "anti_cheat_flags": sc.anti_cheat_flags,
                "bm25_score": sc.bm25_score,
                "interview_questions": [
                    {
                        "category": q.category,
                        "question": q.question,
                        "target": q.target_skill_or_claim,
                        "signal": q.expected_signal,
                    }
                    for q in questions
                ],
            }
        )

    return enriched


@app.get("/")
async def root():
    """Crucible API root providing service discovery and studio connection."""
    return {
        "engine": "Crucible Zero-LLM",
        "status": "online",
        "studio_url": "http://localhost:5173",
        "api": {
            "health": "/api/health",
            "candidates": "/api/candidates",
            "presets": "/api/presets",
            "bias_audit": "/api/jd/bias-audit",
            "compare": "/api/compare",
        },
    }


def recalculate_semantic_similarities():
    """Recalculate semantic_raw and invalidate semantic cache for current ACTIVE_JD."""
    global _SEMANTIC_CACHE
    _SEMANTIC_CACHE.clear()
    try:
        engine = SemanticEngine.get_instance()
        jd_text = (
            f"{ACTIVE_JD.get('title', '')} at {ACTIVE_JD.get('company', '')}. "
            f"Required: {', '.join(ACTIVE_JD.get('required_skills', []))}. "
            f"Nice to have: {', '.join(ACTIVE_JD.get('nice_to_have_skills', []))}."
        )
        jd_emb = engine.encode_text(jd_text)

        cand_texts = [c.get("raw_text") or " ".join(c.get("skills", [])) for c in DEMO_CANDIDATES]
        cand_embs = engine.encode_batch(cand_texts)
        sims = engine.compute_bi_encoder_similarity(cand_embs, jd_emb)
        for i, c in enumerate(DEMO_CANDIDATES):
            c["semantic_raw"] = round(float(sims[i]), 4) if i < len(sims) else 0.5
    except Exception:
        pass


@app.get("/health")
@app.get("/api/health")
async def health():
    """Health check verifying memory state and preloaded candidate count."""
    return {
        "status": "healthy",
        "engine": "Crucible Zero-LLM",
        "active_jd": ACTIVE_JD.get("title"),
        "candidates_count": len(DEMO_CANDIDATES),
    }


class CompareRequest(BaseModel):
    candidate_a_id: str
    candidate_b_id: str


class ParseJDRequest(BaseModel):
    raw_text: str
    title: str | None = None


class SaveJDRequest(BaseModel):
    id: str | None = None
    title: str
    company: str = "Custom Organization"
    min_experience_years: float = 1.0
    required_skills: list[str] = []
    nice_to_have_skills: list[str] = []
    required_education: str = "bachelor"
    required_certifications: list[str] = []
    target_location: str = "Remote"
    required_work_auth: str = "any"
    raw_text: str | None = None


@app.get("/api/candidates")
async def api_candidates(
    mode: str = "hybrid",
    w_kw: float = 35.0,
    w_sem: float = 40.0,
    w_exp: float = 15.0,
    w_edu: float = 5.0,
    w_cert: float = 5.0,
    name_blind: bool = False,
    strict_knockouts: bool = False,
):
    """JSON REST endpoint returning scored and enriched candidates list."""
    candidates = evaluate_candidates(
        mode=mode,
        w_kw=w_kw,
        w_sem=w_sem,
        w_exp=w_exp,
        w_edu=w_edu,
        w_cert=w_cert,
        name_blind=name_blind,
        strict_knockouts=strict_knockouts,
    )
    return {"candidates": candidates, "total": len(candidates), "jd": ACTIVE_JD}


@app.get("/api/candidate/{candidate_id}")
async def api_candidate_detail(candidate_id: str, name_blind: bool = False):
    """JSON REST endpoint returning complete single candidate record."""
    candidates = evaluate_candidates(name_blind=name_blind)
    match = next((c for c in candidates if c["candidate_id"] == candidate_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"candidate": match, "jd": ACTIVE_JD}


@app.get("/api/jd")
async def api_get_jd():
    """Get currently active Job Description."""
    return {"jd": ACTIVE_JD}


@app.get("/api/presets")
async def api_get_presets():
    """Get all pre-configured Job Description presets."""
    return {"presets": PRESET_JDS, "active_id": ACTIVE_JD.get("id")}


@app.post("/api/jd/preset/{preset_id}")
async def api_switch_jd_preset(preset_id: str):
    """Switch active job description preset and recompute rankings."""
    global ACTIVE_JD
    if preset_id not in PRESET_JDS:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")
    ACTIVE_JD = dict(PRESET_JDS[preset_id])
    recalculate_semantic_similarities()
    candidates = evaluate_candidates(mode="hybrid")
    return {"status": "ok", "active_jd": ACTIVE_JD, "candidates": candidates}


@app.post("/api/jd/parse")
async def api_parse_jd(req: ParseJDRequest):
    """Deterministically parse raw JD text/markdown into structured requirements without activating."""
    parsed_jd = parse_job_description(req.raw_text, title=req.title or "Pasted Job Description")
    parsed_jd["raw_text"] = req.raw_text
    return {"status": "ok", "parsed_jd": parsed_jd}


@app.post("/api/jd")
async def api_save_jd(req: SaveJDRequest):
    """Save structured Job Description, set ACTIVE_JD, and recompute rankings."""
    global ACTIVE_JD
    new_jd = {
        "id": req.id or f"custom_jd_{int(time.time())}",
        "title": req.title.strip() or "Custom Role",
        "company": req.company.strip() or "Custom Organization",
        "min_experience_years": float(req.min_experience_years),
        "required_skills": [s.strip() for s in req.required_skills if s.strip()],
        "nice_to_have_skills": [s.strip() for s in req.nice_to_have_skills if s.strip()],
        "required_education": req.required_education,
        "required_certifications": req.required_certifications,
        "target_location": req.target_location,
        "required_work_auth": req.required_work_auth,
    }
    if req.raw_text:
        new_jd["raw_text"] = req.raw_text

    ACTIVE_JD = new_jd
    recalculate_semantic_similarities()
    candidates = evaluate_candidates(mode="hybrid")
    return {"status": "ok", "active_jd": ACTIVE_JD, "candidates": candidates}


@app.post("/api/jd/upload")
async def api_upload_jd(file: UploadFile = File(...)):  # noqa: B008
    """Parse uploaded JD PDF/text, set ACTIVE_JD, and recompute rankings."""
    global ACTIVE_JD
    content = await file.read()
    temp_dir = Path("cache/jd")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / (file.filename or "uploaded_jd.pdf")
    temp_path.write_bytes(content)

    if file.filename and file.filename.lower().endswith(".pdf"):
        doc = parse_document(file_path=temp_path, filename=file.filename)
        jd_text = doc.raw_text
    else:
        jd_text = content.decode("utf-8", errors="ignore")

    parsed_jd = parse_job_description(jd_text, title=file.filename or "Custom Job Description")
    ACTIVE_JD = parsed_jd
    recalculate_semantic_similarities()
    candidates = evaluate_candidates(mode="hybrid")
    return {"status": "ok", "active_jd": ACTIVE_JD, "candidates": candidates}


@app.post("/api/upload-resumes")
async def api_upload_resumes(file: UploadFile = File(...)):  # noqa: B008
    """Parse uploaded resume PDF/text, append to candidates pool, and return updated shortlist."""
    content = await file.read()
    temp_dir = Path("cache/resumes")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / (file.filename or "uploaded_resume.pdf")
    temp_path.write_bytes(content)

    if file.filename and file.filename.lower().endswith(".pdf"):
        doc = parse_document(file_path=temp_path, filename=file.filename)
    else:
        raw_text = content.decode("utf-8", errors="ignore")
        doc = parse_document(raw_text=raw_text, filename=file.filename)

    semantic_engine = SemanticEngine.get_instance()
    emb = semantic_engine.encode_text(doc.raw_text)
    jd_text = (
        f"{ACTIVE_JD['title']} at {ACTIVE_JD['company']}. "
        f"Required: {', '.join(ACTIVE_JD['required_skills'])}. "
        f"Nice to have: {', '.join(ACTIVE_JD.get('nice_to_have_skills', []))}."
    )
    jd_emb = semantic_engine.encode_text(jd_text)
    sims = semantic_engine.compute_bi_encoder_similarity([emb], jd_emb)
    semantic_raw = float(sims[0]) if sims else 0.5

    evidence = (
        next(iter(doc.skill_evidence.values()))
        if doc.skill_evidence
        else (doc.raw_text[:140] if doc.raw_text else None)
    )

    new_cand = {
        "candidate_id": doc.doc_id,
        "name": doc.name,
        "skills": sorted(list(doc.skills)),
        "experience_years": doc.experience_years if doc.experience_years is not None else 0.0,
        "education_level": doc.education_level or "bachelor",
        "certifications": doc.certifications,
        "work_auth": doc.work_auth or "citizen",
        "semantic_raw": semantic_raw,
        "evidence": evidence,
        "raw_text": doc.raw_text,
    }

    existing_idx = next(
        (i for i, c in enumerate(DEMO_CANDIDATES) if c["candidate_id"] == doc.doc_id),
        None,
    )
    if existing_idx is not None:
        DEMO_CANDIDATES[existing_idx] = new_cand
    else:
        DEMO_CANDIDATES.append(new_cand)

    candidates = evaluate_candidates(mode="hybrid")
    return {"status": "ok", "uploaded_id": doc.doc_id, "candidates": candidates}


@app.post("/api/compare")
async def api_compare_candidates(req: CompareRequest):
    """Head-to-head comparison diff matrix between two candidates."""
    candidates = evaluate_candidates()
    cand_a = next((c for c in candidates if c["candidate_id"] == req.candidate_a_id), None)
    cand_b = next((c for c in candidates if c["candidate_id"] == req.candidate_b_id), None)
    if not cand_a or not cand_b:
        raise HTTPException(status_code=404, detail="One or both candidates not found")

    comp = compare_candidates(
        cand_a, cand_b, jd_required_skills=ACTIVE_JD.get("required_skills", [])
    )
    return {
        "candidate_a_id": comp.candidate_a_id,
        "candidate_b_id": comp.candidate_b_id,
        "candidate_a_name": comp.candidate_a_name,
        "candidate_b_name": comp.candidate_b_name,
        "score_delta": comp.score_delta,
        "rank_a": comp.rank_a,
        "rank_b": comp.rank_b,
        "semantic_delta": comp.semantic_delta,
        "keyword_delta": comp.keyword_delta,
        "experience_delta": comp.experience_delta,
        "skills_shared": comp.skills_shared,
        "skills_a_only": comp.skills_a_only,
        "skills_b_only": comp.skills_b_only,
        "summary_verdict": comp.summary_verdict,
        "key_differentiators": comp.key_differentiators,
    }


@app.get("/api/jd/bias-audit")
async def api_bias_audit():
    """Run deterministic audit on active JD for bias, narrow phrasing, and artificial barriers."""
    audit = audit_job_description(ACTIVE_JD)
    return {
        "inclusivity_score": audit.inclusivity_score,
        "total_issues": audit.total_issues,
        "clean_summary": audit.clean_summary,
        "findings": [
            {
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "detected_phrase": f.detected_phrase,
                "explanation": f.explanation,
                "suggested_alternative": f.suggested_alternative,
            }
            for f in audit.findings
        ],
    }


@app.post("/api/export-shortlist")
async def api_export_shortlist():
    """Generate clean Markdown export of current shortlist for hiring manager review."""
    candidates = evaluate_candidates()
    md = [
        f"# Candidate Shortlist: {ACTIVE_JD.get('title')} ({ACTIVE_JD.get('company')})",
        f"**Generated by Crucible Zero-LLM** | **Candidates Evaluated**: {len(candidates)}\n",
        "| Rank | Candidate | Fit Score | Match Status | Matched Skills | Missing Required |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for c in candidates:
        status = (
            "Knocked Out"
            if c.get("is_knocked_out")
            else ("Strong" if c["final_score"] >= 0.75 else "Medium")
        )
        matched = ", ".join(c.get("matched_skills", [])) or "None"
        missing = ", ".join(c.get("missing_skills", [])) or "None"
        md.append(
            f"| #{c['rank']} | {c['name']} | {c['final_score'] * 100:.1f}% | {status} | {matched} | {missing} |"
        )

    return JSONResponse(content={"markdown": "\n".join(md)})
