from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.dense_index import save_dense_index, semantic_product_text
from starter.intent import ShoppingIntent


class FakeEmbedder:
    # Tests use predictable character counts instead of starting Ollama.
    def __init__(self) -> None:
        self.calls = 0
        self.model_name = "nomic-embed-text"

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.calls += 1
        vectors = []
        for value in texts:
            lowered = value.lower()
            raw = (
                float(lowered.count("running") + 1),
                float(lowered.count("winter") + 1),
                float(lowered.count("shoe") + 1),
            )
            length = sum(item * item for item in raw) ** 0.5
            vectors.append(tuple(item / length for item in raw))
        return vectors


class AgentBehaviorTest(unittest.TestCase):
    def setUp(self) -> None:
        # The fake catalog is small, but it still goes through the real SQLite
        # index, reranker, constraints, and clarification policy.
        self.temp_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "SHOE_RED",
                "title": "Red running shoe",
                "categories": ["Clothing", "Shoes"],
                "features": ["lightweight", "breathable"],
                "details": {"color": "red", "size": "9"},
                "price": 55,
                "store": "Example Sports",
                "description": ["shoe for daily road running"],
            },
            {
                "parent_asin": "BOOT_BLUE",
                "title": "Blue winter boot",
                "categories": ["Clothing", "Boots"],
                "features": ["warm", "water resistant"],
                "details": {"color": "blue", "material": "leather"},
                "price": 140,
                "store": "Example Outdoor",
                "description": ["leather boot for cold weather hiking"],
            },
            {
                "parent_asin": "SHOE_BLACK",
                "title": "Black trail shoe",
                "categories": ["Clothing", "Shoes"],
                "features": ["grippy", "water resistant"],
                "details": {"color": "black", "size": "10"},
                "price": 80,
                "store": "Example Outdoor",
                "description": ["shoe for trail running and hiking"],
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.catalog_path = catalog_path
        embedder = FakeEmbedder()
        dense_path = Path(self.temp_directory.name) / "dense_index.npz"
        dense_vectors = embedder.embed([
            f"search_document: {semantic_product_text(product)}"
            for product in products
        ])
        save_dense_index(
            dense_path,
            [product["parent_asin"] for product in products],
            dense_vectors,
            catalog_path,
            embedder.model_name,
        )
        self.agent = Agent(
            catalog_path,
            embedder=embedder,
            dense_index_path=dense_path,
            dense_mode="challenger",
        )

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temp_directory.cleanup()

    def test_sessions_keep_separate_state(self) -> None:
        self.agent.reset("customer_one", {})
        self.agent.reset("customer_two", {})

        self.agent.respond("customer_one", "I need running shoes.", 1, 10)

        self.assertEqual(self.agent._sessions["customer_two"]["intent"].category, "")
        self.assertEqual(self.agent._sessions["customer_one"]["intent"].category, "running shoes")

    def test_verified_default_does_not_require_the_full_dense_index(self) -> None:
        default_agent = Agent(
            self.catalog_path,
            embedder=FakeEmbedder(),
            dense_mode="off",
        )
        try:
            default_agent.reset("default", {})
            response = default_agent.respond("default", "I need shoes.", 1, 2)
        finally:
            default_agent.connection.close()

        self.assertTrue(response["recommendations"])
        self.assertEqual(
            response["decision_trace"]["retrieval"]["strategy"],
            "bm25_plus_nomic",
        )

    def test_router_separates_buying_and_browsing_requests(self) -> None:
        self.agent.reset("buyer", {})
        self.agent.reset("browser", {})

        self.agent.respond(
            "buyer",
            "I'm looking for shoes. A key requirement is: waterproof.",
            1,
            10,
        )
        self.agent.respond("browser", "I'm looking for shoes, but I'm still exploring.", 1, 10)

        self.assertEqual(self.agent._sessions["buyer"]["intent"].route, "buying")
        self.assertEqual(self.agent._sessions["browser"]["intent"].route, "browsing")

    def test_later_answer_updates_the_requested_slot(self) -> None:
        self.agent.reset("customer", {})
        first = self.agent.respond("customer", "I am looking for shoes.", 1, 10)

        self.agent.respond(
            "customer",
            "For that, what matters is: lightweight.",
            2,
            10,
        )

        intent = self.agent._sessions["customer"]["intent"]
        self.assertIn("lightweight", intent.slots[first["ask_attribute"]])

    def test_intent_override_erases_old_preference(self) -> None:
        self.agent.reset("customer", {})
        self.agent.respond("customer", "I'm looking for shoes. I prefer blue.", 1, 10)

        self.agent.respond(
            "customer",
            "Actually, ignore my earlier preference. What I need is: red.",
            2,
            10,
        )

        intent = self.agent._sessions["customer"]["intent"]
        self.assertEqual(intent.category, "shoes")
        self.assertIn("red", intent.slots["color"])
        self.assertNotIn("blue", intent.values())
        blue_revision = next(
            item for item in intent.ledger if "blue" in item.value.lower()
        )
        self.assertEqual("replaced", blue_revision.status)

    def test_selective_replacement_keeps_unrelated_preferences(self) -> None:
        self.agent.reset("customer", {})
        self.agent.respond(
            "customer",
            "I'm looking for shoes. A key requirement is: leather; color: blue.",
            1,
            10,
        )

        response = self.agent.respond(
            "customer",
            "Make it black instead of blue.",
            2,
            10,
        )
        intent = self.agent._sessions["customer"]["intent"]

        self.assertIn("leather", " ".join(intent.slots["material"]))
        self.assertEqual(["black"], intent.slots["color"])
        self.assertNotIn("blue", self.agent._sessions["customer"]["messages"][0].lower())
        self.assertIn(
            "replaced",
            [item.status for item in intent.ledger if "blue" in item.value.lower()],
        )
        self.assertEqual("black", response["decision_trace"]["intent"]["active"]["color"][0])

    def test_no_longer_matters_removes_only_that_attribute(self) -> None:
        self.agent.reset("customer", {})
        self.agent.respond(
            "customer",
            "I'm looking for shoes. A key requirement is: leather; color: blue.",
            1,
            10,
        )

        self.agent.respond("customer", "Color no longer matters.", 2, 10)
        intent = self.agent._sessions["customer"]["intent"]

        self.assertNotIn("color", intent.slots)
        self.assertIn("leather", " ".join(intent.slots["material"]))

    def test_decision_trace_explains_question_value(self) -> None:
        self.agent.reset("customer", {})

        response = self.agent.respond("customer", "I need shoes.", 1, 10)
        clarification = response["decision_trace"]["clarification"]

        self.assertEqual(response["ask_attribute"], clarification["attribute"])
        self.assertGreaterEqual(clarification["expected_candidate_reduction"], 0.0)
        self.assertLessEqual(clarification["expected_candidate_reduction"], 1.0)
        self.assertIn("counterfactual_best_attribute", clarification)

    def test_dense_challenger_keeps_the_incumbent_head(self) -> None:
        intent = ShoppingIntent(category="shoes")
        intent.changed = True
        query = self.agent.embedder.embed(["search_query: winter hiking footwear"])[0]
        dense_results = self.agent.index.dense_index.search(query, limit=3)
        self.agent.index.promotion_margin = -1.0

        promoted = self.agent.index._promote_dense_challenger(
            ["SHOE_RED", "SHOE_BLACK"],
            dense_results,
            query,
            intent,
            set(),
            allow_promotion=True,
        )

        self.assertEqual(promoted[0], "SHOE_RED")
        self.assertEqual(len(promoted), 2)
        self.assertNotEqual(promoted[1], "SHOE_BLACK")

    def test_agent_recommends_while_asking_a_question(self) -> None:
        self.agent.reset("customer", {})

        response = self.agent.respond("customer", "I need a red running shoe.", 1, 10)

        self.assertIsNotNone(response["ask_attribute"])
        self.assertGreater(len(response["recommendations"]), 0)

    def test_agent_does_not_repeat_products_between_turns(self) -> None:
        self.agent.reset("customer", {})

        first = self.agent.respond("customer", "I need shoes.", 1, 1)
        second = self.agent.respond("customer", "I don't have an additional preference.", 2, 1)

        self.assertNotEqual(first["recommendations"], second["recommendations"])

    def test_intent_change_can_reconsider_a_seen_product(self) -> None:
        self.agent.reset("customer", {})

        first = self.agent.respond("customer", "I need a red running shoe.", 1, 1)
        second = self.agent.respond(
            "customer",
            "Actually, ignore my earlier preference. What I need is: red running shoe.",
            2,
            1,
        )

        self.assertEqual(first["recommendations"], second["recommendations"])

    def test_constraints_are_tracked_without_destroying_lexical_recall(self) -> None:
        self.agent.reset("customer", {})

        response = self.agent.respond(
            "customer",
            "I'm looking for footwear. A key requirement is: under $100 without leather.",
            1,
            10,
        )
        intent = self.agent._sessions["customer"]["intent"]

        self.assertEqual(100.0, intent.budget_max)
        self.assertIn("leather", " ".join(intent.exclusions))
        self.assertTrue(response["recommendations"])

    def test_local_agent_reports_zero_model_tokens(self) -> None:
        self.agent.reset("customer", {})

        response = self.agent.respond("customer", "I need shoes.", 1, 10)

        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_semantic_reranking_uses_clean_state_after_a_correction(self) -> None:
        self.agent.reset("customer", {})
        self.agent.respond("customer", "I need blue shoes.", 1, 10)
        calls_before_override = self.agent.embedder.calls

        self.agent.respond(
            "customer",
            "Actually, ignore my earlier preference. What I need is: red shoes.",
            2,
            10,
        )

        self.assertGreater(self.agent.embedder.calls, calls_before_override)
        self.assertEqual(
            "bm25_plus_dense_nomic",
            self.agent.respond(
                "customer",
                "I don't have an additional preference.",
                3,
                10,
            )["decision_trace"]["retrieval"]["strategy"],
        )


if __name__ == "__main__":
    unittest.main()
