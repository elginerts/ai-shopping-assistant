from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "what", "matters", "have", "preference", "those", "options", "not", "quite",
    "right", "yet", "ask", "about", "one", "specific", "attribute", "still",
    "exploring", "additional", "your", "judgment", "use",
}

# The public data shows that feature and material questions reveal useful
# constraints most often. The remaining attributes provide progressively
# narrower refinements.
QUESTION_ORDER = ("feature", "material", "color", "style", "size")


def _text(value: object) -> str:
    # Convert each catalog field into text that can be indexed.
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    # Keep useful search terms in the same order and remove duplicates.
    terms = (
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    )
    return list(dict.fromkeys(terms))


class Agent:
    # The agent keeps separate memory for each conversation and uses BM25
    # retrieval so it can run locally without an external API.

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        # Load the product catalog into an in-memory full-text search index.
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()

        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Start a separate memory store for this customer session.
        self._sessions[session_id] = {
            "messages": [],
            "profile": user_profile,
            "asked_attributes": set(),
            "seen_products": set(),
        }

    def _update_memory(self, session: dict, user_message: str) -> None:
        # Store the latest message and remove an old preference when needed.
        if "ignore my earlier preference" in user_message.lower() and session["messages"]:
            # Keep the product category, but remove the preference that the
            # customer has just replaced.
            category_request = session["messages"][0].split(".", maxsplit=1)[0]
            session["messages"] = [category_request]

            # Earlier questions and recommendations may be useful again now
            # that the customer's requirements have changed.
            session["asked_attributes"].clear()
            session["seen_products"].clear()
        session["messages"].append(user_message)

    def _next_question(self, session: dict) -> str | None:
        # Ask the next useful attribute that has not already been requested.
        for attribute in QUESTION_ORDER:
            if attribute not in session["asked_attributes"]:
                session["asked_attributes"].add(attribute)
                return attribute
        return None

    def _search(self, query: str, top_k: int, seen_products: set[str]) -> list[dict]:
        # Rank products and leave out items already shown in this session.
        unique_terms = _terms(query)[:40]
        if not unique_terms:
            return []

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        # Fetch enough results to replace any products filtered out below.
        search_limit = top_k + len(seen_products)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, search_limit),
        ).fetchall()

        recommendations = [
            {"parent_asin": str(row[0])}
            for row in rows
            if str(row[0]) not in seen_products
        ]
        return recommendations[:top_k]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        # Update the session, retrieve products, and choose a follow-up question.
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        session = self._sessions[session_id]
        self._update_memory(session, user_message)

        # Search the accumulated conversation so later answers refine rather
        # than replace the customer's original request.
        query = " ".join(session["messages"])
        recommendations = self._search(query, top_k, session["seen_products"])
        session["seen_products"].update(
            recommendation["parent_asin"] for recommendation in recommendations
        )
        ask_attribute = self._next_question(session)

        if ask_attribute:
            readable_attribute = ask_attribute.replace("_", " ")
            message = f"Do you have a preference for {readable_attribute}?"
        else:
            message = "Here are the closest matches based on your preferences."

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
