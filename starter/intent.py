from __future__ import annotations

import re
from dataclasses import dataclass, field


ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric", "suede", "denim", "linen", "rubber",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange", "beige", "gold",
    "silver", "navy",
}
USE_CASE_WORDS = {
    "running", "hiking", "walking", "work", "gym", "winter", "summer",
    "outdoor", "travel", "wedding", "casual", "sports", "training",
}

OVERRIDE_MARKERS = (
    "ignore my earlier preference",
    "forget what i said",
    "change my earlier preference",
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;-\t\n")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def classify_attribute(value: str) -> str:
    lowered = value.lower()
    words = set(re.findall(r"[a-z0-9.]+", lowered))
    if "budget" in words or "$" in value or re.search(r"\b(?:under|below|over|above)\s+\d", lowered):
        return "budget"
    if words & MATERIALS:
        return "material"
    if words & COLORS or "color" in words or "colour" in words:
        return "color"
    if re.search(r"\b(?:size|width|wide|narrow|small|medium|large|xl|xxl)\b", lowered):
        return "size"
    if words & USE_CASE_WORDS:
        return "use_case"
    if re.search(r"\b(?:style|fit|sleeve|neck|pattern|department)\b", lowered):
        return "style"
    if re.search(r"\b(?:brand|store|label|designer)\b", lowered):
        return "brand"
    return "feature"


@dataclass
class ShoppingIntent:
    category: str = ""
    slots: dict[str, list[str]] = field(default_factory=dict)
    exclusions: list[str] = field(default_factory=list)
    budget_min: float | None = None
    budget_max: float | None = None
    route: str = "browsing"
    changed: bool = False

    def values(self) -> list[str]:
        return [value for values in self.slots.values() for value in values]

    def query_text(self) -> str:
        return " ".join([self.category, *self.values()]).strip()

    def has_slot(self, attribute: str) -> bool:
        if attribute == "category":
            return bool(self.category)
        if attribute == "budget":
            return self.budget_min is not None or self.budget_max is not None
        return bool(self.slots.get(attribute))


class IntentTracker:
    """Update a compact shopping state from the evaluator's clean text turns."""

    def update(
        self,
        intent: ShoppingIntent,
        user_message: str,
        asked_attribute: str | None,
    ) -> ShoppingIntent:
        lowered = user_message.lower()
        changed = any(marker in lowered for marker in OVERRIDE_MARKERS)
        if lowered.startswith("actually,") and "what i need is" in lowered:
            changed = True

        if changed:
            # The category normally survives an override, while old slots no
            # longer represent what the shopper wants.
            intent.slots.clear()
            intent.exclusions.clear()
            intent.budget_min = None
            intent.budget_max = None

        category = self._extract_category(user_message)
        if category:
            if intent.category and category.lower() != intent.category.lower():
                changed = True
            intent.category = category

        values = self._extract_values(user_message)
        for value in values:
            if self._is_empty_preference(value):
                continue
            is_direct_answer = "for that, what matters is:" in lowered
            attribute = asked_attribute if is_direct_answer and asked_attribute else classify_attribute(value)
            if attribute not in ALLOWED_ATTRIBUTES:
                attribute = "other"
            if attribute == "budget":
                self._update_budget(intent, value)
            else:
                intent.slots[attribute] = _unique([*intent.slots.get(attribute, []), value])[-4:]

        intent.exclusions = _unique([*intent.exclusions, *self._extract_exclusions(user_message)])[-8:]
        self._update_budget(intent, user_message)
        intent.changed = changed
        intent.route = self._choose_route(intent, user_message)
        return intent

    @staticmethod
    def _extract_category(message: str) -> str:
        lowered = message.lower()
        if "what i need is:" in lowered and "looking for" not in lowered:
            return ""
        match = re.search(
            r"\b(?:looking for|need|want)\s+(?:an?\s+|some\s+)?(.+?)(?:[.,;]|\s+but\b|$)",
            message,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        category = _clean(match.group(1))
        category = re.sub(r"^(?:a|an|some)\s+", "", category, flags=re.IGNORECASE)
        return category[:160]

    @staticmethod
    def _extract_values(message: str) -> list[str]:
        markers = (
            "a key requirement is:",
            "what matters is:",
            "what i need is:",
            "preference is:",
        )
        lowered = message.lower()
        for marker in markers:
            position = lowered.find(marker)
            if position >= 0:
                tail = message[position + len(marker):]
                return _unique([_clean(part) for part in tail.split(";")])

        # Intent-override sessions begin with an old preference after the
        # category sentence, so keep that second sentence as a slot value.
        parts = message.split(".", maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            return [_clean(parts[1])]
        return []

    @staticmethod
    def _is_empty_preference(value: str) -> bool:
        lowered = value.lower()
        return not value or any(
            phrase in lowered
            for phrase in (
                "don't have a preference",
                "do not have a preference",
                "no additional preference",
                "use your judgment",
                "not quite right",
                "ask me about",
                "still exploring",
            )
        )

    @staticmethod
    def _extract_exclusions(message: str) -> list[str]:
        exclusions: list[str] = []
        for match in re.finditer(
            r"\b(?:no|without|avoid|except)\s+([a-z][a-z0-9 -]{1,45})",
            message,
            flags=re.IGNORECASE,
        ):
            value = re.split(r"[.,;]", match.group(1), maxsplit=1)[0]
            exclusions.append(_clean(value.lower()))
        return exclusions

    @staticmethod
    def _update_budget(intent: ShoppingIntent, value: str) -> None:
        lowered = value.lower().replace(",", "")
        numbers = [float(item) for item in re.findall(r"\$?\s*(\d+(?:\.\d+)?)", lowered)]
        if not numbers:
            return
        if re.search(r"\b(?:under|below|less than|up to|max(?:imum)?)\b", lowered):
            intent.budget_max = numbers[-1]
        elif re.search(r"\b(?:over|above|more than|min(?:imum)?)\b", lowered):
            intent.budget_min = numbers[-1]
        elif "budget around" in lowered or "budget" in lowered:
            intent.budget_max = numbers[-1] * 1.15

    @staticmethod
    def _choose_route(intent: ShoppingIntent, message: str) -> str:
        lowered = message.lower()
        if "still exploring" in lowered and not intent.values():
            return "browsing"
        hard_slots = sum(
            len(intent.slots.get(name, []))
            for name in ("material", "color", "size", "brand", "feature", "use_case")
        )
        has_budget = intent.budget_min is not None or intent.budget_max is not None
        return "buying" if hard_slots or has_budget or intent.exclusions else "browsing"
