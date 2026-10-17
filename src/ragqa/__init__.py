"""rag-doc-qa: a document Q&A service with citations.

Retrieval (BM25 + dense vectors, RRF-fused, cross-encoder re-ranked) with a
grounded generation layer and an abstention gate, plus an evaluation harness.

Import order matters for logging only; the package has no import side effects
beyond defining names, so `import ragqa` is cheap and safe in tests.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build_pipeline",
    "Pipeline",
    "Settings",
    "get_settings",
]


def __getattr__(name: str):  # lazy re-exports keep `import ragqa` dependency-free
    if name in ("build_pipeline", "Pipeline"):
        from . import pipeline

        return getattr(pipeline, name)
    if name in ("Settings", "get_settings"):
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
