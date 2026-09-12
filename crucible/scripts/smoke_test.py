import time

from crucible.core.explanation import generate_narrative_explanation
from crucible.core.keyword import compute_keyword_match
from crucible.core.normalizer import compute_spearman_validation
from crucible.core.parser import parse_document
from crucible.core.scorecard import compute_scorecards
from crucible.core.semantic import SemanticEngine


def run_smoke_test():
    print("=" * 60)
    print("CRUCIBLE SMOKE TEST — ZERO-LLM SHORTLISTING PIPELINE")
    print("=" * 60)

    start_total = time.perf_counter()

    # 1. Document parsing check
    raw_doc = (
        "Devon Miller\n"
        "Junior Developer\n"
        "Experience: 2+ years building Python microservices with FastAPI and PostgreSQL.\n"
        "Configured CI/CD with Docker on AWS."
    )
    vocab = {"Python", "FastAPI", "PostgreSQL", "Docker", "CI/CD", "AWS"}
    parsed = parse_document(raw_text=raw_doc, filename="Devon_Miller.pdf", skill_vocab=vocab)
    assert parsed.name == "Devon Miller"
    assert parsed.experience_years == 2.0
    assert "Python" in parsed.skills
    print("[PASS] Resilient deterministic extraction verified.")

    # 2. Keyword coverage check
    req = {"Python", "FastAPI", "PostgreSQL", "Docker", "React"}
    nice = {"AWS", "Kubernetes"}
    kw = compute_keyword_match(parsed.skills, req, nice)
    assert 0.0 < kw.keyword_score <= 1.0
    print(f"[PASS] Keyword coverage computed: {kw.keyword_score * 100:.1f}%")

    # 3. Two-stage semantic retrieval check
    engine = SemanticEngine.get_instance()
    jd_text = "Junior Full Stack Developer skilled in Python, FastAPI, and PostgreSQL."
    jd_emb = engine.encode_text(jd_text)
    cand_emb = engine.encode_text(parsed.raw_text)
    sim = engine.compute_bi_encoder_similarity([cand_emb], jd_emb)[0]
    assert 0.50 <= sim <= 1.0
    print(f"[PASS] Local CPU semantic similarity: {sim:.4f}")

    # 4. Scorecard & Normalization check
    cands = [
        {
            "candidate_id": "c1",
            "name": parsed.name,
            "keyword_score": kw.keyword_score,
            "semantic_raw": sim,
            "experience_years": parsed.experience_years,
        },
        {
            "candidate_id": "c2",
            "name": "Irrelevant Candidate",
            "keyword_score": 0.0,
            "semantic_raw": 0.35,
            "experience_years": None,
        },
    ]
    cards = compute_scorecards(cands, min_years=1.0)
    assert cards[0].rank == 1 and cards[0].candidate_id == "c1"
    print(f"[PASS] Scorecard ranked candidate #1 (Final score: {cards[0].final_score * 100:.1f})")

    # 5. Grounded Narrative Explanation check
    quote = parsed.skill_evidence.get("Python", "")
    verdict = generate_narrative_explanation(
        parsed.name,
        kw.matched_required,
        kw.missing_required,
        quote,
    )
    assert "Devon Miller is a strong match" in verdict

    print(f'[PASS] Grounded verdict synthesized:\n  "{verdict}"')

    # 6. Spearman validation check
    manual = [3.0, 1.0]
    system = [cards[0].final_score, cards[1].final_score]
    rho, p_val = compute_spearman_validation(manual, system)
    assert rho > 0.0
    print(f"[PASS] Spearman validation rank correlation: rho={rho:.2f} (p={p_val:.3f})")

    elapsed = (time.perf_counter() - start_total) * 1000
    print("-" * 60)
    print(f"ALL SMOKE TESTS PASSED IN {elapsed:.1f}ms — ZERO-LLM INVARIANTS INTACT")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
