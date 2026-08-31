from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from starter.config import AgentConfig


class AgentConfigTest(unittest.TestCase):
    def test_defaults_match_the_verified_runtime(self) -> None:
        # An empty environment should reproduce the settings used for scoring.
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig.from_environment()

        self.assertEqual("off", config.dense_mode)
        self.assertEqual(0.18, config.semantic_weight)
        self.assertEqual(16, config.rerank_limit)
        self.assertEqual("guarded", config.question_policy)
        self.assertFalse(config.emit_decision_trace)

    def test_explicit_dense_mode_wins_over_environment(self) -> None:
        # Tests and experiments need a reliable way to override shell settings.
        with patch.dict(os.environ, {"THREADLINE_DENSE_MODE": "learned"}, clear=True):
            config = AgentConfig.from_environment(dense_mode="off")

        self.assertEqual("off", config.dense_mode)

    def test_invalid_correction_mode_fails_early(self) -> None:
        # Invalid settings should stop during setup, not halfway through a session.
        with patch.dict(
            os.environ,
            {"THREADLINE_CORRECTION_SEMANTIC": "unknown"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                AgentConfig.from_environment()

    def test_invalid_trace_setting_fails_early(self) -> None:
        # Only explicit boolean values should control extra debug output.
        with patch.dict(
            os.environ,
            {"THREADLINE_DECISION_TRACE": "sometimes"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                AgentConfig.from_environment()


if __name__ == "__main__":
    unittest.main()
