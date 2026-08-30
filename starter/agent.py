from __future__ import annotations

import os
from pathlib import Path

from starter.dialogue import ClarificationPolicy
from starter.intent import IntentTracker, ShoppingIntent
from starter.ollama_embeddings import EmbeddingCache, OllamaEmbeddingClient
from starter.retrieval import CatalogIndex


class Agent:
    """Local multi-turn shopping agent used by the official evaluator."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        embedder: OllamaEmbeddingClient | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        supplied_embedder = embedder is not None
        self.embedder = embedder or OllamaEmbeddingClient()
        embedding_cache = None
        if not supplied_embedder:
            cache_directory = Path(os.getenv("THREADLINE_CACHE_DIR", ".threadline_cache"))
            cache_directory.mkdir(parents=True, exist_ok=True)
            embedding_cache = EmbeddingCache(
                str(cache_directory / "product_embeddings.sqlite3"),
                self.embedder.model_name,
            )
        self.index = CatalogIndex(
            self.catalog_path,
            self.embedder,
            semantic_weight=float(os.getenv("THREADLINE_SEMANTIC_WEIGHT", "0.18")),
            rerank_limit=int(os.getenv("THREADLINE_RERANK_LIMIT", "16")),
            embedding_cache=embedding_cache,
        )
        self.connection = self.index.connection
        self.intent_tracker = IntentTracker()
        self.clarification_policy = ClarificationPolicy(
            mode=os.getenv("THREADLINE_QUESTION_POLICY", "guarded")
        )
        self.correction_semantic_mode = os.getenv(
            "THREADLINE_CORRECTION_SEMANTIC", "clean"
        )
        if self.correction_semantic_mode not in {"lexical", "clean"}:
            raise ValueError(
                "THREADLINE_CORRECTION_SEMANTIC must be 'lexical' or 'clean'"
            )
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
            "use_semantic_reranker": True,
            "last_diagnostic": {},
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
            turn,
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
            # Do not embed the raw correction beside the old request. Rebuild
            # the query from active ledger slots so retired values stay out.
            session["use_semantic_reranker"] = (
                self.correction_semantic_mode == "clean"
            )
            if self.correction_semantic_mode == "clean":
                session["messages"] = [intent.query_text()]
            elif "ignore my earlier preference" in lowered_message:
                category_request = session["messages"][0].split(".", maxsplit=1)[0]
                session["messages"] = [category_request, user_message]
            else:
                session["messages"] = [intent.query_text()]

        if intent.selective_revision:
            # Recompile from active slots so replaced words cannot remain in
            # the raw conversation query after a partial correction.
            session["messages"] = [intent.query_text()]
            session["seen_products"].clear()
            session["asked_attributes"].clear()
            session["use_semantic_reranker"] = (
                self.correction_semantic_mode == "clean"
            )

        seen_before = set(session["seen_products"])
        recommendations, candidate_ids, retrieval_evidence = self.index.search(
            intent=intent,
            profile=session["profile"],
            seen_products=session["seen_products"],
            top_k=top_k,
            conversation_query=" ".join(session["messages"]),
            use_semantic_reranker=session["use_semantic_reranker"],
        )

        question_decision = self.clarification_policy.choose(
            intent=intent,
            candidate_ids=candidate_ids,
            products=self.index.products,
            asked_attributes=session["asked_attributes"],
            adaptive=session["adaptive_questions"],
        )
        ask_attribute = question_decision.attribute
        if ask_attribute:
            session["asked_attributes"].add(ask_attribute)
        session["last_asked_attribute"] = ask_attribute
        session["last_diagnostic"] = {
            **retrieval_evidence,
            "seen_before": sorted(seen_before),
            "ask_attribute": ask_attribute,
        }
        session["seen_products"].update(
            item["parent_asin"] for item in recommendations
        )

        route_label = "specific request" if intent.route == "buying" else "browsing request"
        message = f"I’m treating this as a {route_label}. {question_decision.question}"
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "decision_trace": {
                "intent": {
                    "route": intent.route,
                    "changed_this_turn": intent.changed,
                    "selective_revision_this_turn": intent.selective_revision,
                    "active": intent.active_state(),
                    "history": [
                        {
                            "revision_id": item.revision_id,
                            "attribute": item.attribute,
                            "value": item.value,
                            "status": item.status,
                            "turn": item.turn,
                            "replaced_by": item.replaced_by,
                            "ended_turn": item.ended_turn,
                        }
                        for item in intent.ledger[-12:]
                    ],
                },
                "retrieval": {
                    "strategy": (
                        "bm25_plus_nomic"
                        if session["use_semantic_reranker"]
                        else "lexical_after_correction"
                    ),
                    "candidate_count": len(candidate_ids),
                    "returned_count": len(recommendations),
                },
                "clarification": question_decision.trace(),
            },
        }

    def diagnostic_snapshot(self, session_id: str) -> dict:
        """Return stage evidence for the latest turn without exposing it in the API."""
        if session_id not in self._sessions:
            return {}
        diagnostic = self._sessions[session_id].get("last_diagnostic", {})
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in diagnostic.items()
        }
