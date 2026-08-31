from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "dense_similarity",
    "dense_reciprocal_rank",
    "bm25_reciprocal_rank",
    "nomic_reciprocal_rank",
    "constraint_coverage",
    "category_coverage",
    "valid_for_constraints",
    "buying_route",
    "revision_turn",
)


@dataclass(frozen=True, slots=True)
class PromotionCandidate:
    """One grounded candidate and the numeric evidence used to rank it."""

    parent_asin: str
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PromotionModel:
    """A compact linear ranker that can be inspected and reproduced."""

    weights: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    margin: float = 0.0

    def score(self, features: tuple[float, ...]) -> float:
        # Standardising here keeps features with different units comparable.
        values = np.asarray(features, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        return float(((values - means) / scales) @ weights)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        # JSON is portable and lets judges see exactly what the small model uses.
        payload = {
            "version": 1,
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "means": list(self.means),
            "scales": list(self.scales),
            "margin": self.margin,
            "metadata": metadata or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PromotionModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Promotion model feature schema does not match this code version")
        vectors = (payload["weights"], payload["means"], payload["scales"])
        if any(len(vector) != len(FEATURE_NAMES) for vector in vectors):
            raise ValueError("Promotion model has the wrong number of features")
        if any(float(scale) <= 0.0 for scale in payload["scales"]):
            raise ValueError("Promotion model scales must be positive")
        return cls(
            weights=tuple(float(value) for value in payload["weights"]),
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
            margin=float(payload.get("margin", 0.0)),
        )


def fit_pairwise_ranker(
    positive: list[tuple[float, ...]],
    negative: list[tuple[float, ...]],
    epochs: int = 500,
    learning_rate: float = 0.05,
    regularization: float = 0.02,
) -> PromotionModel:
    """Fit target-minus-distractor pairs with deterministic logistic descent."""

    if not positive or len(positive) != len(negative):
        raise ValueError("Training needs the same non-zero number of positive and negative rows")
    differences = np.asarray(positive, dtype=np.float64) - np.asarray(negative, dtype=np.float64)
    scales = differences.std(axis=0)
    scales[scales < 1e-6] = 1.0
    training = differences / scales
    weights = np.zeros(training.shape[1], dtype=np.float64)

    # Every row means the first product should outrank the second. Clipping
    # prevents overflow without adding a heavyweight machine-learning library.
    for _ in range(epochs):
        logits = np.clip(training @ weights, -30.0, 30.0)
        errors = 1.0 / (1.0 + np.exp(logits))
        gradient = -(training.T @ errors) / len(training) + regularization * weights
        weights -= learning_rate * gradient

    # Runtime scores use absolute feature rows, so train-time centring is kept
    # only for scale. A shared offset cannot change pairwise ordering.
    return PromotionModel(
        weights=tuple(float(value) for value in weights),
        means=tuple(0.0 for _ in FEATURE_NAMES),
        scales=tuple(float(value) for value in scales),
    )
