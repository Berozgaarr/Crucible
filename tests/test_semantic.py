"""Tests for crucible.core.semantic."""

import time

import numpy as np

from crucible.core.semantic import SemanticEngine


def test_semantic_engine_singleton():
    engine1 = SemanticEngine.get_instance()
    engine2 = SemanticEngine.get_instance()
    assert engine1 is engine2


def test_bi_encoder_dimension():
    engine = SemanticEngine.get_instance()
    emb = engine.encode_text("Python backend engineer with FastAPI and Docker experience.")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)
    assert emb.dtype == np.float32

    # Unit norm assertion (normalize_embeddings=True)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-3


def test_bi_encoder_batch_and_similarity():
    engine = SemanticEngine.get_instance()
    jd = "Seeking Full Stack Developer proficient in React, Node.js, and PostgreSQL."
    jd_emb = engine.encode_text(jd)

    candidates = [
        "Full stack engineer skilled in React, Node.js, and relational databases like PostgreSQL.",
        "Experienced Python data scientist working on pandas, scikit-learn, and machine learning.",
        "Professional classical pianist and composer performing classical music symphonies.",
    ]
    cand_embs = engine.encode_batch(candidates)
    assert cand_embs.shape == (3, 384)

    scores = engine.compute_bi_encoder_similarity(cand_embs, jd_emb)
    assert len(scores) == 3

    # Candidate 0 (React/Node/Postgres) must be top match
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]
    assert scores[0] > 0.60


def test_cross_encoder_rerank_top_k_bound():
    engine = SemanticEngine.get_instance()
    jd = "Junior software developer internship building REST APIs with Python."
    candidates = [
        f"Software engineering candidate number {i} with Python background" for i in range(12)
    ]

    # Strictly test top-8 bounding
    top_8_scores = engine.rerank_top_k(jd, candidates, top_k=8)
    assert len(top_8_scores) == 8

    # Empty check
    assert engine.rerank_top_k(jd, [], top_k=8) == []


def test_cpu_batch_latency():
    engine = SemanticEngine.get_instance()
    jd = "Python software engineer with cloud and database experience."
    jd_emb = engine.encode_text(jd)

    # 18 candidates representing the full hackathon batch
    mock_batch = [
        f"Candidate {i}: Software engineer with Python, APIs, and DBs for {i + 1} years."
        for i in range(18)
    ]

    start = time.perf_counter()
    embs = engine.encode_batch(mock_batch)
    _ = engine.compute_bi_encoder_similarity(embs, jd_emb)
    duration_ms = (time.perf_counter() - start) * 1000

    assert embs.shape == (18, 384)
    # CPU latency check: 18 resumes should encode smoothly
    assert duration_ms < 2500  # Generous safety envelope for initial CPU warm run
