"""Offline cache generator.

Precomputes extraction, embeddings, and scorecards for instant demo readiness.
"""

import json
from pathlib import Path

from crucible.config import DEFAULT_JD, DEMO_CANDIDATES
from crucible.core.explanation import explain_candidate
from crucible.core.keyword import compute_keyword_match
from crucible.core.scorecard import compute_scorecards
from crucible.core.semantic import SemanticEngine


def generate_cache(output_dir: Path | str = "cache") -> Path:
    """Precompute embeddings, scorecards, and explanations, saving to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating Crucible offline pre-cache...")
    engine = SemanticEngine.get_instance()

    # Pre-encode JD
    jd_summary = (
        f"{DEFAULT_JD['title']} at {DEFAULT_JD['company']}. "
        f"Requires {', '.join(DEFAULT_JD['required_skills'])}. "
        f"Nice to have: {', '.join(DEFAULT_JD.get('nice_to_have_skills', []))}."
    )
    jd_emb = engine.encode_text(jd_summary)

    processed_candidates = []
    texts = [c["raw_text"] for c in DEMO_CANDIDATES]
    cand_embs = engine.encode_batch(texts)
    sim_scores = engine.compute_bi_encoder_similarity(cand_embs, jd_emb)

    for i, c in enumerate(DEMO_CANDIDATES):
        cand_copy = dict(c)
        cand_copy["semantic_raw"] = sim_scores[i]
        kw_res = compute_keyword_match(
            c["skills"],
            DEFAULT_JD["required_skills"],
            DEFAULT_JD.get("nice_to_have_skills", []),
        )
        cand_copy["keyword_score"] = kw_res.keyword_score
        cand_copy["matched_skills"] = kw_res.matched_required
        cand_copy["missing_skills"] = kw_res.missing_required
        processed_candidates.append(cand_copy)

    # Compute scorecards
    cards = compute_scorecards(
        processed_candidates,
        min_years=DEFAULT_JD.get("min_experience_years"),
    )
    card_dict = {sc.candidate_id: sc for sc in cards}

    final_payload = {
        "jd": DEFAULT_JD,
        "candidates": [],
    }

    for c in processed_candidates:
        sc = card_dict[c["candidate_id"]]
        expl = explain_candidate(
            candidate_id=c["candidate_id"],
            name=c["name"],
            matched_required=c["matched_skills"],
            missing_required=c["missing_skills"],
            top_evidence_quote=c.get("evidence"),
        )
        final_payload["candidates"].append(
            {
                "candidate_id": c["candidate_id"],
                "name": c["name"],
                "rank": sc.rank,
                "keyword_score": sc.keyword_score,
                "semantic_raw": sc.semantic_raw,
                "semantic_norm": sc.semantic_norm,
                "experience_score": sc.experience_score,
                "final_score": sc.final_score,
                "experience_years": c.get("experience_years"),
                "matched_skills": c["matched_skills"],
                "missing_skills": c["missing_skills"],
                "evidence": c.get("evidence"),
                "narrative": expl.narrative,
            }
        )

    # Sort descending by final score
    final_payload["candidates"].sort(key=lambda x: x["rank"])

    cache_file = out_path / "demo_cache.json"
    with open(cache_file, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"Successfully cached {len(processed_candidates)} candidates to {cache_file}")
    return cache_file


if __name__ == "__main__":
    generate_cache()
