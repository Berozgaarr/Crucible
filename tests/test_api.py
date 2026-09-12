from fastapi.testclient import TestClient

from crucible.app import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_jd" in data


def test_api_candidates_and_weights():
    res = client.get("/api/candidates?mode=hybrid&w_kw=50&w_sem=50")
    assert res.status_code == 200
    data = res.json()
    assert "candidates" in data
    assert len(data["candidates"]) >= 10
    first = data["candidates"][0]
    assert "rank" in first
    assert "final_score" in first
    assert "candidate_id" in first


def test_api_candidate_detail():
    res = client.get("/api/candidate/cand_01")
    assert res.status_code == 200
    data = res.json()
    assert data["candidate"]["candidate_id"] == "cand_01"
    assert "interview_questions" in data["candidate"]


def test_api_candidate_detail_not_found():
    res = client.get("/api/candidate/non_existent_id")
    assert res.status_code == 404


def test_api_presets_and_switch():
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()["presets"]
    assert "fullstack_junior" in presets

    switch_res = client.post("/api/jd/preset/backend_senior")
    assert switch_res.status_code == 200
    assert switch_res.json()["active_jd"]["id"] == "backend_senior"

    # Reset back to fullstack_junior
    client.post("/api/jd/preset/fullstack_junior")


def test_api_compare():
    payload = {"candidate_a_id": "cand_01", "candidate_b_id": "cand_04"}
    res = client.post("/api/compare", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["candidate_a_id"] == "cand_01"
    assert data["candidate_b_id"] == "cand_04"
    assert "summary_verdict" in data
    assert "key_differentiators" in data


def test_api_bias_audit():
    res = client.get("/api/jd/bias-audit")
    assert res.status_code == 200
    data = res.json()
    assert "inclusivity_score" in data
    assert "findings" in data


def test_api_export_shortlist():
    res = client.post("/api/export-shortlist")
    assert res.status_code == 200
    data = res.json()
    assert "markdown" in data
    assert "# Candidate Shortlist" in data["markdown"]


def test_api_root():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "Crucible Zero-LLM"
    assert "studio_url" in data


def test_api_upload_resumes():
    content = (
        "Alice Smith\n"
        "Experience: Senior Backend Engineer with 5 years experience in Python and FastAPI.\n"
        "Skills: Python, FastAPI, Docker, PostgreSQL, Redis, Kubernetes.\n"
    )
    files = {"file": ("alice_smith.txt", content.encode("utf-8"), "text/plain")}
    res = client.post("/api/upload-resumes", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert any(c["name"] == "Alice Smith" for c in data["candidates"])


def test_api_upload_jd():
    jd_content = (
        "Principal Database Architect\n"
        "Requirements: 6+ years experience, Bachelor degree.\n"
        "Must have SQL, PostgreSQL, Database skills.\n"
    )
    files = {"file": ("custom_jd.txt", jd_content.encode("utf-8"), "text/plain")}
    res = client.post("/api/jd/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["active_jd"]["title"] == "custom_jd.txt"


def test_api_parse_jd():
    raw_text = (
        "Staff Site Reliability Engineer\n"
        "Requirements:\n"
        "- 5+ years of experience with Kubernetes, Docker, and Python\n"
        "- Master's degree in Computer Science preferred\n"
        "- US Citizen only / clearance required\n"
        "- AWS Certified or CKA certification is a plus\n"
    )
    res = client.post("/api/jd/parse", json={"raw_text": raw_text, "title": "Staff SRE"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    parsed = data["parsed_jd"]
    assert parsed["title"] == "Staff SRE"
    assert parsed["min_experience_years"] == 5.0
    assert parsed["required_education"] in ["master", "bachelor"]
    assert parsed["required_work_auth"] == "citizen"
    assert "Docker" in parsed["required_skills"] or "Kubernetes" in parsed["required_skills"]


def test_api_save_jd():
    payload = {
        "title": "Lead Cloud Infrastructure Architect",
        "company": "Nimbus Systems",
        "min_experience_years": 4.0,
        "required_skills": ["AWS", "Docker", "Kubernetes", "Git"],
        "nice_to_have_skills": ["Python", "Terraform"],
        "required_education": "bachelor",
        "required_certifications": ["AWS Certified"],
        "target_location": "Remote",
        "required_work_auth": "any",
        "raw_text": "Lead Cloud Architect position requiring AWS and Kubernetes.",
    }
    res = client.post("/api/jd", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["active_jd"]["title"] == "Lead Cloud Infrastructure Architect"
    assert data["active_jd"]["company"] == "Nimbus Systems"
    assert len(data["candidates"]) > 0
    assert data["candidates"][0]["rank"] == 1
