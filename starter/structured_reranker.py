from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from starter.intent import ShoppingIntent


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class RankableProduct(Protocol):
    """The small product view needed by the structured reranker."""

    corpus: str
    price: float | None


@dataclass(frozen=True, slots=True)
class ConstraintScore:
    """Explainable evidence for one product in the final slate."""

    coverage: float
    exact_matches: int
    contradictions: int


def _tokens(value: str) -> tuple[str, ...]:
    # Short words add noise in product descriptions, so the reranker ignores
    # them unless they contain a number such as a size or model value.
    return tuple(
        token.lower()
        for token in TOKEN_RE.findall(value)
        if len(token) > 1 or token.isdigit()
    )


def active_constraints(intent: ShoppingIntent) -> tuple[str, ...]:
    """Return shopper requirements without treating the category as a slot."""

    return tuple(value for value in intent.values() if _tokens(value))


def score_product(product: RankableProduct, intent: ShoppingIntent) -> ConstraintScore:
    """Measure explicit matches and contradictions against the active ledger."""

    constraints = active_constraints(intent)
    matched_total = 0.0
    exact_matches = 0
    for value in constraints:
        value_tokens = _tokens(value)
        hits = sum(
            bool(re.search(rf"\b{re.escape(token)}\b", product.corpus))
            for token in value_tokens
        )
        matched_total += hits / len(value_tokens)
        exact_matches += int(hits == len(value_tokens))

    contradictions = 0
    for excluded in intent.exclusions:
        excluded_tokens = _tokens(excluded)
        if excluded_tokens and all(
            re.search(rf"\b{re.escape(token)}\b", product.corpus)
            for token in excluded_tokens
        ):
            contradictions += 1
    if intent.budget_min is not None and product.price is not None:
        contradictions += int(product.price < intent.budget_min)
    if intent.budget_max is not None and product.price is not None:
        contradictions += int(product.price > intent.budget_max)

    coverage = matched_total / len(constraints) if constraints else 0.0
    return ConstraintScore(coverage, exact_matches, contradictions)


def rerank_top_ten(
    ranked_ids: list[str],
    products: dict[str, RankableProduct],
    intent: ShoppingIntent,
    minimum_gain: float = 0.30,
) -> list[str]:
    """Promote a clear exact match while leaving uncertain rankings untouched."""

    constraints = active_constraints(intent)
    if len(ranked_ids) < 2 or len(constraints) < 2:
        return ranked_ids

    evidence = {item: score_product(products[item], intent) for item in ranked_ids}
    head = ranked_ids[0]
    eligible = [
        item for item in ranked_ids
        if evidence[item].contradictions == 0 and evidence[item].exact_matches >= 2
    ]
    if not eligible:
        return ranked_ids

    # Stable max and stable sorting keep BM25/Nomic order for tied products.
    best = max(eligible, key=lambda item: evidence[item].coverage)
    gain = evidence[best].coverage - evidence[head].coverage
    if best == head or gain < minimum_gain:
        return ranked_ids
    return [best, *(item for item in ranked_ids if item != best)]
