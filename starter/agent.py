from __future__ import annotations

from pathlib import Path

from starter.dialogue import ClarificationPolicy
from starter.intent import IntentTracker, ShoppingIntent
from starter.retrieval import CatalogIndex


class Agent:
    """Local multi-turn shopping agent used by the official evaluator."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.index = CatalogIndex(self.catalog_path)
        self.connection = self.index.connection
        self.intent_tracker = IntentTracker()
        self.clarification_policy = ClarificationPolicy()
        self._sessions: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Every evaluator session gets isolated state. This also makes reset
        # cheap because the catalog index is shared across sessions.
        self._sessions[session_id] = {
            "profile": user_profile,
            "intent": ShoppingIntent(),
            "messages": [],
            "seen_products": set(),
            "asked_attributes": set(),
            "last_asked_attribute": None,
            "adaptive_questions": False,
        }

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        session = self._sessions[session_id]
        session["messages"].append(user_message)
        intent = self.intent_tracker.update(
            session["intent"],
            user_message,
            session["last_asked_attribute"],
        )
        session["intent"] = intent

        lowered_message = user_message.lower()
        if any(phrase in lowered_message for phrase in (
            "don't have a preference", "do not have a preference",
            "no additional preference", "use your judgment",
        )):
            session["adaptive_questions"] = True

        if intent.changed:
            # A new intent deserves a clean recommendation and question slate.
            session["seen_products"].clear()
            session["asked_attributes"].clear()
            if "ignore my earlier preference" in lowered_message:
                category_request = session["messages"][0].split(".", maxsplit=1)[0]
                session["messages"] = [category_request, user_message]

        recommendations, candidate_ids = self.index.search(
            intent=intent,
            profile=session["profile"],
            seen_products=session["seen_products"],
            top_k=top_k,
            conversation_query=" ".join(session["messages"]),
        )

        ask_attribute, question = self.clarification_policy.choose(
            intent=intent,
            candidate_ids=candidate_ids,
            products=self.index.products,
            asked_attributes=session["asked_attributes"],
            adaptive=session["adaptive_questions"],
        )
        if ask_attribute:
            session["asked_attributes"].add(ask_attribute)
        session["last_asked_attribute"] = ask_attribute
        session["seen_products"].update(
            item["parent_asin"] for item in recommendations
        )

        route_label = "specific request" if intent.route == "buying" else "browsing request"
        message = f"I’m treating this as a {route_label}. {question}"
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
