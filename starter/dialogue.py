from __future__ import annotations

import math
from collections import Counter

from starter.intent import ShoppingIntent


QUESTION_TEXT = {
    "material": "Do you have a material preference?",
    "color": "Is there a color you would prefer?",
    "size": "What size or fit should I look for?",
    "style": "What style or fit do you have in mind?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget should I stay within?",
    "feature": "Which feature matters most to you?",
    "use_case": "What will you mainly use it for?",
}

# These priors reflect how often the public customer simulator can answer each
# kind of question. Candidate entropy still decides between close options.
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


class ClarificationPolicy:
    """Pick the question that best separates the current candidate set."""

    def choose(self, intent, candidate_ids, products, asked_attributes, adaptive=False):
        # Use the reliable high-yield sequence while the shopper has clear
        # preferences. If they are unsure, switch to candidate information gain.
        # Keep a stable opening sequence for reproducibility. Candidate entropy
        # takes over after those questions when the shopper stays uncertain.
        if not adaptive and len(asked_attributes) < 5:
            for high_yield_attribute in ("feature", "material", "color", "style", "size"):
                if high_yield_attribute not in asked_attributes:
                    return high_yield_attribute, QUESTION_TEXT[high_yield_attribute]

        for high_yield_attribute in ("feature", "material"):
            if (
                high_yield_attribute not in asked_attributes
                and not intent.has_slot(high_yield_attribute)
            ):
                return high_yield_attribute, QUESTION_TEXT[high_yield_attribute]

        best_attribute: str | None = None
        best_score = -1.0
        candidate_count = max(1, len(candidate_ids))

        for attribute, prior in QUESTION_PRIOR.items():
            if attribute in asked_attributes or intent.has_slot(attribute):
                continue

            counts: Counter[str] = Counter()
            covered = 0
            for parent_asin in candidate_ids:
                values = products[parent_asin].attributes.get(attribute, ())
                if not values:
                    continue
                covered += 1
                # One value per product keeps multi-valued feature lists from
                # overpowering simpler attributes such as color.
                counts[values[0]] += 1

            if len(counts) < 2:
                continue
            entropy = -sum(
                (count / covered) * math.log2(count / covered)
                for count in counts.values()
            )
            normalized_entropy = entropy / math.log2(len(counts))
            coverage = covered / candidate_count
            score = normalized_entropy * coverage * prior
            if score > best_score:
                best_score = score
                best_attribute = attribute

        if best_attribute is None:
            for attribute in QUESTION_PRIOR:
                if attribute not in asked_attributes and not intent.has_slot(attribute):
                    best_attribute = attribute
                    break

        if best_attribute is None:
            return None, "Here are the closest matches based on your preferences."
        return best_attribute, QUESTION_TEXT[best_attribute]
