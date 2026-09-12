"""Pure Python / NumPy BM25Okapi implementation for zero-LLM lexical search."""

import math
import re
from collections import Counter
from collections.abc import Sequence


class BM25Index:
    """In-memory BM25Okapi index over a corpus of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.term_freqs: list[dict[str, int]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Normalize and tokenize text into lowercase word tokens."""
        if not text:
            return []
        # Support technical tokens like c++, c#, .net
        tokens = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
        return [t.strip(".") for t in tokens if t.strip(".")]

    def fit(self, corpus: Sequence[str]) -> "BM25Index":
        """Build BM25 parameters across candidate documents."""
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return self

        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        total_len = 0
        for doc in corpus:
            tokens = self.tokenize(doc)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

        # Compute standard Lucene/BM25 Okapi IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # BM25 standard IDF formulation with smoothing
            val = (self.corpus_size - freq + 0.5) / (freq + 0.5)
            self.idf[term] = math.log(1.0 + max(val, 1e-6))

        return self

    def score(self, query: str) -> list[float]:
        """Score all indexed documents against query string."""
        if self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]

            for doc_idx, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
                )
                scores[doc_idx] += idf_val * (numerator / denominator)

        return [round(s, 4) for s in scores]
