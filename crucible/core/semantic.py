"""Two-stage local neural semantic retrieval engine using sentence-transformers on CPU."""

import threading
from typing import ClassVar

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


class SemanticEngine:
    """Thread-safe singleton for local neural semantic embeddings and cross-encoder reranking."""

    _instance: ClassVar["SemanticEngine | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        bi_model_name: str = "all-MiniLM-L6-v2",
        cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.bi_model_name = bi_model_name
        self.cross_model_name = cross_model_name
        self._bi_model: SentenceTransformer | None = None
        self._cross_model: CrossEncoder | None = None

    @classmethod
    def get_instance(
        cls,
        bi_model_name: str = "all-MiniLM-L6-v2",
        cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> "SemanticEngine":
        """Get or initialize singleton SemanticEngine instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(
                    bi_model_name=bi_model_name,
                    cross_model_name=cross_model_name,
                    device=device,
                )
            return cls._instance

    @property
    def bi_model(self) -> SentenceTransformer:
        """Lazy load bi-encoder model on CPU."""
        if self._bi_model is None:
            self._bi_model = SentenceTransformer(self.bi_model_name, device=self.device)
        return self._bi_model

    @property
    def cross_model(self) -> CrossEncoder:
        """Lazy load cross-encoder model on CPU."""
        if self._cross_model is None:
            self._cross_model = CrossEncoder(self.cross_model_name, device=self.device)
        return self._cross_model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode single string to 384-d normalized float32 vector."""
        if not text.strip():
            return np.zeros(384, dtype=np.float32)
        emb = self.bi_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Batch encode strings into 2D float32 ndarray."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embs = self.bi_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(embs, dtype=np.float32)

    def compute_bi_encoder_similarity(
        self,
        candidate_embeddings: list[np.ndarray] | np.ndarray,
        jd_embedding: np.ndarray,
    ) -> list[float]:
        """Compute cosine similarity between candidates and JD."""
        if len(candidate_embeddings) == 0:
            return []

        cands = np.asarray(candidate_embeddings, dtype=np.float32)
        jd = np.asarray(jd_embedding, dtype=np.float32)

        # Cosine similarity for unit-normalized vectors is dot product
        similarities = np.dot(cands, jd)
        return [round(float(s), 4) for s in similarities]

    def compute_hierarchical_similarity(
        self,
        candidate_sections_list: list[dict[str, str]],
        candidate_full_texts: list[str],
        jd_text: str,
        section_weights: dict[str, float] | None = None,
    ) -> list[float]:
        """Hierarchical section-aware semantic matching against JD."""
        if not candidate_full_texts:
            return []

        weights = section_weights or {
            "experience": 0.40,
            "projects": 0.30,
            "skills": 0.15,
            "full_text": 0.15,
        }

        jd_emb = self.encode_text(jd_text)

        full_embs = self.encode_batch(candidate_full_texts)

        # Batch encode non-empty sections to minimize model forward passes
        exp_texts = [
            (
                candidate_sections_list[i].get("experience", "")
                if i < len(candidate_sections_list)
                else ""
            )
            for i in range(len(candidate_full_texts))
        ]
        proj_texts = [
            (
                candidate_sections_list[i].get("projects", "")
                if i < len(candidate_sections_list)
                else ""
            )
            for i in range(len(candidate_full_texts))
        ]
        skills_texts = [
            (
                candidate_sections_list[i].get("skills", "")
                if i < len(candidate_sections_list)
                else ""
            )
            for i in range(len(candidate_full_texts))
        ]

        exp_embs = self.encode_batch([t for t in exp_texts if t])
        proj_embs = self.encode_batch([t for t in proj_texts if t])
        skills_embs = self.encode_batch([t for t in skills_texts if t])

        exp_iter = iter(exp_embs)
        proj_iter = iter(proj_embs)
        skills_iter = iter(skills_embs)

        scores: list[float] = []
        for i in range(len(candidate_full_texts)):
            full_emb = full_embs[i]
            sim_full = float(np.dot(full_emb, jd_emb))

            sim_exp = float(np.dot(next(exp_iter), jd_emb)) if exp_texts[i] else sim_full
            sim_proj = float(np.dot(next(proj_iter), jd_emb)) if proj_texts[i] else sim_full
            sim_skills = float(np.dot(next(skills_iter), jd_emb)) if skills_texts[i] else sim_full

            hierarchical_score = (
                (weights["experience"] * sim_exp)
                + (weights["projects"] * sim_proj)
                + (weights["skills"] * sim_skills)
                + (weights["full_text"] * sim_full)
            )
            scores.append(round(float(hierarchical_score), 4))

        return scores

    def rerank_top_k(
        self,
        jd_text: str,
        candidate_texts: list[str],
        top_k: int = 8,
    ) -> list[float]:
        """Rerank top-k candidates jointly using cross-encoder with sigmoid probability scaling."""
        if not candidate_texts:
            return []

        eval_texts = candidate_texts[:top_k]
        pairs = [[jd_text, cand[:1500]] for cand in eval_texts]
        scores = self.cross_model.predict(pairs)

        # Apply numerically stable sigmoid to map cross-encoder logits to [0.0, 1.0]
        sigmoid_scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float32)))
        return [round(float(s), 4) for s in sigmoid_scores]

    def find_top_evidence_sentences(
        self,
        jd_clause: str,
        candidate_sentences: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Find candidate sentences with highest cross-encoder relevance to specific JD clause."""
        clean_sentences = [s.strip() for s in candidate_sentences if len(s.strip()) > 15][:20]
        if not clean_sentences or not jd_clause.strip():
            return []

        pairs = [[jd_clause, sent] for sent in clean_sentences]
        scores = self.cross_model.predict(pairs)

        scored = list(zip(clean_sentences, [float(s) for s in scores], strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(sent, round(score, 4)) for sent, score in scored[:top_k]]
