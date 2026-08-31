from __future__ import annotations

import unittest
from dataclasses import dataclass

from starter.intent import ShoppingIntent
from starter.structured_reranker import rerank_top_ten


@dataclass
class Product:
    corpus: str
    price: float | None = None


class StructuredRerankerTest(unittest.TestCase):
    def test_promotes_a_clear_multi_constraint_match(self) -> None:
        intent = ShoppingIntent(slots={"color": ["red"], "material": ["leather"]})
        products = {
            "first": Product("red canvas shoe"),
            "best": Product("red leather shoe"),
            "other": Product("black leather shoe"),
        }

        ranked = rerank_top_ten(["first", "best", "other"], products, intent)

        self.assertEqual("best", ranked[0])

    def test_keeps_original_order_when_evidence_is_weak(self) -> None:
        intent = ShoppingIntent(slots={"color": ["red"]})
        products = {
            "first": Product("canvas shoe"),
            "second": Product("red shoe"),
        }

        ranked = rerank_top_ten(["first", "second"], products, intent)

        self.assertEqual(["first", "second"], ranked)

    def test_exclusion_blocks_promotion(self) -> None:
        intent = ShoppingIntent(
            slots={"color": ["red"], "material": ["leather"]},
            exclusions=["metal"],
        )
        products = {
            "first": Product("red canvas shoe"),
            "blocked": Product("red leather shoe with metal buckle"),
        }

        ranked = rerank_top_ten(["first", "blocked"], products, intent)

        self.assertEqual("first", ranked[0])


if __name__ == "__main__":
    unittest.main()
