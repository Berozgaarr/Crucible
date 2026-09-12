"""Deterministic technical interview question generator citing candidate claims and skill gaps."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class InterviewQuestion:
    category: str  # claim_verification, gap_probe, architectural_tradeoff
    question: str
    target_skill_or_claim: str
    expected_signal: str


# Deterministic questions for common skill gaps
GAP_PROBES: dict[str, dict[str, str]] = {
    "Kubernetes": {
        "question": "Candidate lacks Kubernetes. In your previous environments, how did you handle service discovery, rolling deployments, and container orchestration?",
        "expected_signal": "Understanding of pod lifecycle, ingress controllers, or alternative orchestration like Nomad/ECS.",
    },
    "Docker": {
        "question": "Candidate has not listed Docker. Walk through how you achieve consistent runtime environments between local development and production systems.",
        "expected_signal": "Familiarity with containerization fundamentals, cgroups, namespaces, or Nix/chroot.",
    },
    "PostgreSQL": {
        "question": "Candidate lacks explicit PostgreSQL experience. How do you design indexing strategies and analyze query plans (EXPLAIN ANALYZE) for slow queries?",
        "expected_signal": "B-Tree vs GIN/GiST indexes, transaction isolation levels, and vacuuming.",
    },
    "Redis": {
        "question": "Candidate lacks Redis. When caching high-throughput data, what eviction policies and cache invalidation patterns (cache-aside vs write-through) do you rely on?",
        "expected_signal": "TTL strategies, memory bounds, stampede prevention.",
    },
    "FastAPI": {
        "question": "Candidate lacks FastAPI. How do you implement asynchronous I/O request handling, dependency injection, and automatic schema validation in your backends?",
        "expected_signal": "Event loop concurrency, Pydantic/dataclass serialization, ASGI vs WSGI.",
    },
    "AWS": {
        "question": "Candidate lacks AWS experience. How have you architected multi-region failovers, IAM role least-privilege policies, and cost-efficient cloud storage?",
        "expected_signal": "IAM boundaries, cloud network topology (VPC, peering), object storage lifecycle.",
    },
}


def generate_interview_questions(
    name: str,
    matched_skills: Sequence[str],
    missing_skills: Sequence[str],
    evidence_quote: str | None = None,
    detected_metrics: Sequence[str] | None = None,
) -> list[InterviewQuestion]:
    """Deterministically synthesize probing technical interview questions for candidate."""
    questions: list[InterviewQuestion] = []

    # 1. Claim Verification Probe (if verbatim quote or metrics are available)
    if evidence_quote:
        questions.append(
            InterviewQuestion(
                category="claim_verification",
                question=f'In your resume, you cited: "{evidence_quote.strip()}". Can you walk through the architectural trade-offs, roadblocks, and specific tools you chose during this initiative?',
                target_skill_or_claim="Verbatim Resume Evidence",
                expected_signal="Technical depth, verification that candidate was hands-on rather than passive observer.",
            )
        )

    # 2. Metric Verification Probe
    if detected_metrics:
        top_metric = detected_metrics[0]
        questions.append(
            InterviewQuestion(
                category="claim_verification",
                question=f'You highlighted a measured impact of "{top_metric}". How was this metric measured and baselined, and what was the direct causal mechanism behind the improvement?',
                target_skill_or_claim=top_metric,
                expected_signal="Observability rigor, telemetry tools used (Prometheus, Grafana, Datadog), and analytical clarity.",
            )
        )

    # 3. Gap Probes (Target primary missing requirements)
    for missing in missing_skills[:2]:
        if missing in GAP_PROBES:
            info = GAP_PROBES[missing]
            questions.append(
                InterviewQuestion(
                    category="gap_probe",
                    question=info["question"],
                    target_skill_or_claim=missing,
                    expected_signal=info["expected_signal"],
                )
            )
        else:
            questions.append(
                InterviewQuestion(
                    category="gap_probe",
                    question=f"The job requires hands-on experience with {missing}. What equivalent tools or fundamental paradigms have you worked with that transfer to {missing}?",
                    target_skill_or_claim=missing,
                    expected_signal=f"Grounded conceptual transferability to {missing}.",
                )
            )

    # 4. Core Strength Architectural Probe
    if matched_skills:
        core_skill = matched_skills[0]
        questions.append(
            InterviewQuestion(
                category="architectural_tradeoff",
                question=f"You have demonstrated proficiency in {core_skill}. What are the biggest performance bottlenecks or footguns you've experienced in production with {core_skill}, and how did you resolve them?",
                target_skill_or_claim=core_skill,
                expected_signal="Senior-level production battle scars, failure modes, and debugging methodology.",
            )
        )

    return questions
