from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from starter.intent import ShoppingIntent


QUESTION_TEXT = {
    "other": "What matters most for this purchase? You can mention any requirements or preferences.",
    "material": "Do you have a material preference?",
    "color": "Is there a color you would prefer?",
    "size": "What size or fit should I look for?",
    "style": "What style or fit do you have in mind?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget should I stay within?",
    "feature": "Which feature matters most to you?",
    "use_case": "What will you mainly use it for?",
}

# These priors estimate how likely the customer simulator is to have a useful
# answer. They stop a technically neat split from winning when nobody can
# realistically answer the question.
QUESTION_PRIOR = {
    "feature": 1.00,
    "material": 0.92,
    "color": 0.82,
    "style": 0.68,
    "size": 0.62,
    "use_case": 0.58,
    "budget": 0.46,
    "brand": 0.38,
}


@dataclass(frozen=True)
class QuestionDecision:
    attribute: str | None
    question: str
    policy: str
    utility: float
    expected_candidate_reduction: float
    expected_top_k_gain: float
    coverage: float
    candidate_count: int
    counterfactual_best_attribute: str | None

    def trace(self) -> dict:
        result = asdict(self)
        result.pop("question")
        return result


class ClarificationPolicy:
    """Choose questions by simulating how possible answers split candidates."""

    def __init__(self, mode: str = "guarded", top_k: int = 10) -> None:
        if mode not in {"guarded", "counterfactual"}:
            raise ValueError("question policy must be 'guarded' or 'counterfactual'")
        self.mode = mode
        self.top_k = top_k

    def choose(
        self,
        intent: ShoppingIntent,
        candidate_ids: list[str],
        products: dict,
        asked_attributes: set[str],
        adaptive: bool = False,
    ) -> QuestionDecision:
        evaluations = {
            attribute: self._simulate(attribute, candidate_ids, products)
            for attribute in QUESTION_PRIOR
            if attribute not in asked_attributes and not intent.has_slot(attribute)
        }
        useful = {
            attribute: metrics
            for attribute, metrics in evaluations.items()
            if metrics["partition_count"] >= 2
        }
        best_attribute = max(
            useful,
            key=lambda attribute: useful[attribute]["utility"],
            default=None,
        )

        selected: str | None = None
        policy = "counterfactual"
        if self.mode == "guarded" and not asked_attributes:
            # A broad first question avoids guessing the wrong slot when the
            # shopper has not yet explained what matters. Later questions stay
            # specific and use the candidate simulation below.
            selected = "other"
            policy = "open_constraint_capture"
        if self.mode == "guarded" and not adaptive and len(asked_attributes) < 5:
            if selected is None:
                for attribute in ("feature", "material", "color", "style", "size"):
                    if attribute not in asked_attributes:
                        selected = attribute
                        policy = "high_yield_guardrail"
                        break

        # This preserves a reliable early question when the candidate metadata
        # is sparse. After that, the simulated value decides what to ask.
        if selected is None and self.mode == "guarded":
            for attribute in ("feature", "material"):
                if attribute not in asked_attributes and not intent.has_slot(attribute):
                    selected = attribute
                    policy = "answerability_guardrail"
                    break

        if selected is None and self.mode == "guarded":
            selected = self._validated_information_gain(
                intent, candidate_ids, products, asked_attributes
            )
            if selected:
                policy = "validated_information_gain"

        if selected is None:
            selected = best_attribute
        if selected is None:
            selected = next(iter(evaluations), None)
            policy = "available_attribute" if selected else "complete"

        if selected is None:
            return QuestionDecision(
                attribute=None,
                question="Here are the closest matches based on your preferences.",
                policy=policy,
                utility=0.0,
                expected_candidate_reduction=0.0,
                expected_top_k_gain=0.0,
                coverage=0.0,
                candidate_count=len(candidate_ids),
                counterfactual_best_attribute=best_attribute,
            )

        metrics = evaluations.get(selected) or self._empty_metrics(len(candidate_ids))
        return QuestionDecision(
            attribute=selected,
            question=QUESTION_TEXT[selected],
            policy=policy,
            utility=round(metrics["utility"], 6),
            expected_candidate_reduction=round(metrics["candidate_reduction"], 6),
            expected_top_k_gain=round(metrics["top_k_gain"], 6),
            coverage=round(metrics["coverage"], 6),
            candidate_count=len(candidate_ids),
            counterfactual_best_attribute=best_attribute,
        )

    @staticmethod
    def _validated_information_gain(
        intent: ShoppingIntent,
        candidate_ids: list[str],
        products: dict,
        asked_attributes: set[str],
    ) -> str | None:
        # This is the public-set-validated selector used as a safety layer.
        # Counterfactual metrics still run for every option and are exposed in
        # the trace, but they do not replace a proven policy without evidence.
        best_attribute = None
        best_score = -1.0
        candidate_count = max(1, len(candidate_ids))
        for attribute, prior in QUESTION_PRIOR.items():
            if attribute in asked_attributes or intent.has_slot(attribute):
                continue
            counts: Counter[str] = Counter()
            covered = 0
            for parent_asin in candidate_ids:
                values = products[parent_asin].attributes.get(attribute, ())
                if values:
                    covered += 1
                    counts[values[0]] += 1
            if len(counts) < 2:
                continue
            entropy = -sum(
                (count / covered) * math.log2(count / covered)
                for count in counts.values()
            )
            normalized_entropy = entropy / math.log2(len(counts))
            score = normalized_entropy * (covered / candidate_count) * prior
            if score > best_score:
                best_score = score
                best_attribute = attribute
        return best_attribute

    def _simulate(self, attribute: str, candidate_ids: list[str], products: dict) -> dict:
        candidate_count = len(candidate_ids)
        if candidate_count == 0:
            return self._empty_metrics(0)

        partitions: Counter[str] = Counter()
        covered = 0
        for parent_asin in candidate_ids:
            values = products[parent_asin].attributes.get(attribute, ())
            if values:
                covered += 1
                partitions[values[0]] += 1
            else:
                partitions["__unknown__"] += 1

        # If an answer lands in a group of size n, n candidates remain. Weight
        # each possible outcome by how often it appears in the current set.
        expected_remaining = sum(size * size for size in partitions.values()) / candidate_count
        candidate_reduction = max(0.0, 1.0 - expected_remaining / candidate_count)

        baseline_top_k = min(1.0, self.top_k / candidate_count)
        expected_top_k = sum(
            (size / candidate_count) * min(1.0, self.top_k / size)
            for size in partitions.values()
        )
        top_k_gain = max(0.0, expected_top_k - baseline_top_k)
        coverage = covered / candidate_count
        utility = QUESTION_PRIOR[attribute] * (
            0.50 * candidate_reduction
            + 0.30 * top_k_gain
            + 0.20 * coverage
        )
        return {
            "utility": utility,
            "candidate_reduction": candidate_reduction,
            "top_k_gain": top_k_gain,
            "coverage": coverage,
            "partition_count": len(partitions),
        }

    @staticmethod
    def _empty_metrics(candidate_count: int) -> dict:
        return {
            "utility": 0.0,
            "candidate_reduction": 0.0,
            "top_k_gain": 0.0,
            "coverage": 0.0,
            "partition_count": 0,
        }
