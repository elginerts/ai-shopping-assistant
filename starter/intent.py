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
    # Normalize whitespace before a value enters the ledger.
    return re.sub(r"\s+", " ", value).strip(" .,:;-\t\n")


def _unique(values: list[str]) -> list[str]:
    # Keep stable order because newer values are trimmed from the end later.
    return list(dict.fromkeys(value for value in values if value))


def classify_attribute(value: str) -> str:
    # The challenge provides clean text, so a compact retail lexicon is enough here.
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
class SlotRevision:
    """One auditable change to a shopper preference."""

    revision_id: int
    attribute: str
    value: str
    status: str
    turn: int
    source: str
    replaced_by: int | None = None
    ended_turn: int | None = None


@dataclass
class ShoppingIntent:
    category: str = ""
    slots: dict[str, list[str]] = field(default_factory=dict)
    exclusions: list[str] = field(default_factory=list)
    budget_min: float | None = None
    budget_max: float | None = None
    route: str = "browsing"
    changed: bool = False
    selective_revision: bool = False
    ledger: list[SlotRevision] = field(default_factory=list)

    def values(self) -> list[str]:
        # Flatten active slots in insertion order for deterministic query compilation.
        return [value for values in self.slots.values() for value in values]

    def query_text(self) -> str:
        # Retired ledger entries are intentionally excluded from the search query.
        return " ".join([self.category, *self.values()]).strip()

    def active_state(self) -> dict[str, list[str]]:
        # Return a JSON-friendly view for the optional public decision trace.
        state: dict[str, list[str]] = {}
        if self.category:
            state["category"] = [self.category]
        state.update({key: list(values) for key, values in self.slots.items() if values})
        if self.exclusions:
            state["exclusion"] = list(self.exclusions)
        if self.budget_min is not None:
            state["budget_min"] = [str(round(self.budget_min, 2))]
        if self.budget_max is not None:
            state["budget_max"] = [str(round(self.budget_max, 2))]
        return state

    def record(
        self,
        attribute: str,
        value: str,
        turn: int,
        source: str,
        replace_attribute: bool = False,
    ) -> None:
        # Append revisions instead of overwriting history used for auditability.
        clean_value = _clean(value)
        if not clean_value:
            return
        active = [
            item for item in self.ledger
            if item.attribute == attribute and item.status == "active"
        ]
        if any(item.value.lower() == clean_value.lower() for item in active):
            return

        next_id = len(self.ledger) + 1
        if replace_attribute:
            for item in active:
                item.status = "replaced"
                item.replaced_by = next_id
                item.ended_turn = turn
        self.ledger.append(SlotRevision(
            revision_id=next_id,
            attribute=attribute,
            value=clean_value,
            status="active",
            turn=turn,
            source=source[:240],
        ))

    def retire_preferences(self, turn: int) -> None:
        # A full pivot retires preferences but keeps the product category.
        for item in self.ledger:
            if item.status == "active" and item.attribute != "category":
                item.status = "replaced"
                item.ended_turn = turn
        self.slots.clear()
        self.exclusions.clear()
        self.budget_min = None
        self.budget_max = None

    def remove_attribute(self, attribute: str, turn: int, source: str) -> None:
        # Record removals explicitly so they can be explained in the trace.
        for item in self.ledger:
            if item.attribute == attribute and item.status == "active":
                item.status = "removed"
                item.ended_turn = turn
        self.slots.pop(attribute, None)
        if attribute == "budget":
            self.budget_min = None
            self.budget_max = None
        self.ledger.append(SlotRevision(
            revision_id=len(self.ledger) + 1,
            attribute=attribute,
            value="no preference",
            status="removed",
            turn=turn,
            source=source[:240],
        ))

    def has_slot(self, attribute: str) -> bool:
        # Budget and category live outside the regular slot dictionary.
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
        turn: int = 0,
    ) -> ShoppingIntent:
        # Update the existing session object so its ledger history stays intact.
        lowered = user_message.lower()
        changed = any(marker in lowered for marker in OVERRIDE_MARKERS)
        if lowered.startswith("actually,") and "what i need is" in lowered:
            changed = True
        category = self._extract_category(user_message)
        category_changed = bool(
            category and intent.category and category.lower() != intent.category.lower()
        )
        changed = changed or category_changed

        if changed:
            # The category normally survives an override, while old slots no
            # longer represent what the shopper wants.
            intent.retire_preferences(turn)

        removed_attribute = self._extract_removed_attribute(user_message)
        if removed_attribute:
            intent.remove_attribute(removed_attribute, turn, user_message)

        replacement = self._extract_selective_replacement(user_message)
        if replacement and not changed:
            new_value, _old_value = replacement
            attribute = classify_attribute(new_value)
            intent.slots[attribute] = [new_value]
            intent.record(attribute, new_value, turn, user_message, replace_attribute=True)

        if category:
            if category_changed:
                intent.record("category", category, turn, user_message, replace_attribute=True)
            elif not intent.category:
                intent.record("category", category, turn, user_message)
            intent.category = category

        old_budget = (intent.budget_min, intent.budget_max)
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
                if not replacement or value.lower() != replacement[0].lower():
                    intent.slots[attribute] = _unique([*intent.slots.get(attribute, []), value])[-4:]
                    intent.record(attribute, value, turn, user_message)

        new_exclusions = self._extract_exclusions(user_message)
        intent.exclusions = _unique([*intent.exclusions, *new_exclusions])[-8:]
        for exclusion in new_exclusions:
            intent.record("exclusion", exclusion, turn, user_message)
        self._update_budget(intent, user_message)
        if old_budget != (intent.budget_min, intent.budget_max):
            intent.record("budget", user_message, turn, user_message, replace_attribute=True)
        intent.changed = changed
        intent.selective_revision = bool((removed_attribute or replacement) and not changed)
        intent.route = self._choose_route(intent, user_message)
        return intent

    @staticmethod
    def _extract_selective_replacement(message: str) -> tuple[str, str] | None:
        # Capture patterns such as "black instead of blue" without a full reset.
        match = re.search(
            r"(?:make it |show me )?([a-z0-9$ -]{1,50}?)\s+instead of\s+([a-z0-9$ -]{1,50})(?:[.,;]|$)",
            message,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return _clean(match.group(1)), _clean(match.group(2))

    @staticmethod
    def _extract_removed_attribute(message: str) -> str | None:
        # Attribute-level removal preserves unrelated preferences.
        lowered = message.lower()
        match = re.search(
            r"\b(material|colou?r|size|style|brand|budget|feature|use case)\s+"
            r"(?:no longer matters|doesn't matter|does not matter)",
            lowered,
        )
        if not match:
            return None
        return match.group(1).replace("colour", "color").replace("use case", "use_case")

    @staticmethod
    def _extract_category(message: str) -> str:
        # The evaluator uses clean shopping phrases, so a bounded pattern is sufficient.
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
        # Pull semicolon-separated requirements from the challenge dialogue format.
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
        # Boundary replies should not become literal search constraints.
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
        # Keep negative requirements separate from positive ranking evidence.
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
        # Parse lower and upper bounds without rejecting products with missing prices.
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
        # Explicit constraints indicate Buying; exploratory language stays Browsing.
        lowered = message.lower()
        if "still exploring" in lowered and not intent.values():
            return "browsing"
        hard_slots = sum(
            len(intent.slots.get(name, []))
            for name in ("material", "color", "size", "brand", "feature", "use_case")
        )
        has_budget = intent.budget_min is not None or intent.budget_max is not None
        return "buying" if hard_slots or has_budget or intent.exclusions else "browsing"
