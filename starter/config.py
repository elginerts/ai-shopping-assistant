from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


VALID_DENSE_MODES = {"off", "challenger", "learned"}
VALID_CORRECTION_MODES = {"lexical", "clean"}


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Runtime settings collected in one place before components are built."""

    dense_mode: str
    dense_index_path: Path
    cache_directory: Path
    semantic_weight: float
    rerank_limit: int
    promotion_margin: float
    promotion_model_path: Path
    question_policy: str
    correction_semantic_mode: str

    @classmethod
    def from_environment(
        cls,
        dense_mode: str | None = None,
        dense_index_path: str | Path | None = None,
    ) -> "AgentConfig":
        # Explicit constructor arguments take priority, which keeps tests and
        # evaluator experiments independent from the user's shell settings.
        selected_dense_mode = dense_mode or os.getenv("THREADLINE_DENSE_MODE", "off")
        if selected_dense_mode not in VALID_DENSE_MODES:
            raise ValueError(
                "THREADLINE_DENSE_MODE must be 'off', 'challenger', or 'learned'"
            )

        correction_mode = os.getenv("THREADLINE_CORRECTION_SEMANTIC", "clean")
        if correction_mode not in VALID_CORRECTION_MODES:
            raise ValueError(
                "THREADLINE_CORRECTION_SEMANTIC must be 'lexical' or 'clean'"
            )

        return cls(
            dense_mode=selected_dense_mode,
            dense_index_path=Path(
                dense_index_path
                or os.getenv(
                    "THREADLINE_DENSE_INDEX",
                    ".threadline_cache/dense_index.npz",
                )
            ),
            cache_directory=Path(
                os.getenv("THREADLINE_CACHE_DIR", ".threadline_cache")
            ),
            semantic_weight=float(os.getenv("THREADLINE_SEMANTIC_WEIGHT", "0.18")),
            rerank_limit=int(os.getenv("THREADLINE_RERANK_LIMIT", "16")),
            promotion_margin=float(os.getenv("THREADLINE_PROMOTION_MARGIN", "0.03")),
            promotion_model_path=Path(
                os.getenv("THREADLINE_PROMOTION_MODEL", "models/promotion_model.json")
            ),
            question_policy=os.getenv("THREADLINE_QUESTION_POLICY", "guarded"),
            correction_semantic_mode=correction_mode,
        )
