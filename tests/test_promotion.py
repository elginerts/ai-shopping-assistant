from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from starter.promotion import FEATURE_NAMES, PromotionModel, fit_pairwise_ranker


class PromotionModelTest(unittest.TestCase):
    def test_pairwise_training_prefers_the_positive_pattern(self) -> None:
        # The synthetic rows make coverage the only useful signal.
        positive = [(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0)] * 20
        negative = [(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)] * 20
        model = fit_pairwise_ranker(positive, negative, epochs=80)
        self.assertGreater(model.score(positive[0]), model.score(negative[0]))

    def test_json_round_trip_keeps_the_feature_schema(self) -> None:
        model = PromotionModel(
            weights=tuple(0.1 for _ in FEATURE_NAMES),
            means=tuple(0.0 for _ in FEATURE_NAMES),
            scales=tuple(1.0 for _ in FEATURE_NAMES),
            margin=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path, {"purpose": "test"})
            restored = PromotionModel.load(path)
        self.assertEqual(model, restored)


if __name__ == "__main__":
    unittest.main()
