"""Tests for pure Python/NumPy BM25Okapi and Reciprocal Rank Fusion."""

from crucible.core.bm25 import BM25Index
from crucible.core.normalizer import reciprocal_rank_fusion


def test_bm25_tokenization_and_scoring():
    corpus = [
        "Senior Python and FastAPI backend developer building PostgreSQL APIs with Docker.",
        "Frontend specialist in React, TypeScript, and Tailwind CSS design systems.",
        "DevOps platform engineer handling Kubernetes, Terraform, AWS, and CI/CD pipelines.",
    ]
    index = BM25Index().fit(corpus)

    # Search for backend tech
    scores = index.score("Python FastAPI PostgreSQL")
    assert len(scores) == 3
    # Candidate 0 must rank highest for Python FastAPI query
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_empty_query_or_corpus():
    index = BM25Index().fit([])
    assert index.score("python") == []

    index_with_docs = BM25Index().fit(["doc one", "doc two"])
    assert index_with_docs.score("") == [0.0, 0.0]


def test_reciprocal_rank_fusion():
    bm25_ranks = ["cand_01", "cand_02", "cand_03"]
    dense_ranks = ["cand_02", "cand_01", "cand_04"]

    rrf = reciprocal_rank_fusion([bm25_ranks, dense_ranks], k=60)

    # Both cand_01 and cand_02 appear in top-2 of both lists, should have highest scores
    assert rrf["cand_01"] > rrf["cand_03"]
    assert rrf["cand_02"] > rrf["cand_04"]
    assert max(rrf.values()) == 1.0  # normalized max
