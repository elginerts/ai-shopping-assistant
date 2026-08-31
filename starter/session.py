from __future__ import annotations

from typing import Any, TypedDict

from starter.intent import ShoppingIntent
from starter.promotion import PromotionCandidate


class SessionState(TypedDict):
    """Mutable state kept separately for each evaluator session."""

    profile: dict[str, Any]
    intent: ShoppingIntent
    messages: list[str]
    seen_products: set[str]
    asked_attributes: set[str]
    last_asked_attribute: str | None
    adaptive_questions: bool
    use_semantic_reranker: bool
    last_diagnostic: dict[str, Any]
    last_promotion_candidates: tuple[PromotionCandidate, ...]
