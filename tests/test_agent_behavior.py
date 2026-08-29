from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


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
        self.agent = Agent(catalog_path)

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temp_directory.cleanup()

    def test_sessions_keep_separate_state(self) -> None:
        self.agent.reset("customer_one", {})
        self.agent.reset("customer_two", {})

        self.agent.respond("customer_one", "I need running shoes.", 1, 10)

        self.assertEqual(self.agent._sessions["customer_two"]["intent"].category, "")
        self.assertEqual(self.agent._sessions["customer_one"]["intent"].category, "running shoes")

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


if __name__ == "__main__":
    unittest.main()
