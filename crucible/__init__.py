"""Crucible deterministic zero-LLM smart shortlisting engine."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
