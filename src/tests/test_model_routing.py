"""
Unit tests for model routing and LLM client.

Tests:
  - Routing returns correct model per tier
  - Fallback triggers on simulated provider error
  - ZDR header present on every request
"""

import unittest
from unittest.mock import patch, MagicMock

from src.agents.model_router import (
    ModelConfig,
    get_research_model,
    get_performance_model,
    get_alert_model,
)
from src.agents.llm_client import call, LLMClientError, _build_headers


class TestResearchRouting(unittest.TestCase):

    def test_deep_tier_returns_current_high_effort_model(self):
        config = get_research_model("deep")
        assert config is not None
        self.assertEqual(config.model, "anthropic/claude-sonnet-4-6")
        self.assertEqual(config.effort, "high")
        self.assertFalse(config.extended_thinking)
        self.assertEqual(config.fallback_model, "moonshotai/kimi-k2.5")

    def test_standard_tier_returns_sonnet(self):
        config = get_research_model("standard")
        assert config is not None
        self.assertEqual(config.model, "anthropic/claude-sonnet-4-6")
        self.assertEqual(config.effort, "medium")
        self.assertFalse(config.extended_thinking)
        self.assertEqual(config.fallback_model, "moonshotai/kimi-k2.5")

    def test_skip_tier_returns_none(self):
        config = get_research_model("skip")
        self.assertIsNone(config)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            get_research_model("nonexistent")

    def test_case_insensitive(self):
        config = get_research_model("Deep")
        assert config is not None
        self.assertEqual(config.model, "anthropic/claude-sonnet-4-6")


class TestPerformanceRouting(unittest.TestCase):

    def test_returns_kimi(self):
        config = get_performance_model()
        self.assertEqual(config.model, "moonshotai/kimi-k2.5")
        self.assertFalse(config.extended_thinking)

    def test_has_fallback(self):
        config = get_performance_model()
        self.assertEqual(config.fallback_model, "anthropic/claude-haiku-4-5-20251001")


class TestAlertRouting(unittest.TestCase):

    def test_returns_kimi(self):
        config = get_alert_model()
        self.assertEqual(config.model, "moonshotai/kimi-k2.5")
        self.assertFalse(config.extended_thinking)

    def test_has_fallback(self):
        config = get_alert_model()
        self.assertEqual(config.fallback_model, "anthropic/claude-haiku-4-5-20251001")


class TestZDRHeaders(unittest.TestCase):

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_no_store_header_present(self):
        # Re-import to pick up patched env
        from src.agents import llm_client
        llm_client.OPENROUTER_API_KEY = "test-key"
        headers = _build_headers()
        self.assertEqual(headers["X-OpenRouter-No-Store"], "true")

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_auth_header_present(self):
        from src.agents import llm_client
        llm_client.OPENROUTER_API_KEY = "test-key"
        headers = _build_headers()
        self.assertEqual(headers["Authorization"], "Bearer test-key")


class TestFallbackBehavior(unittest.TestCase):

    @patch("src.agents.llm_client._call_openrouter")
    @patch("src.agents.llm_client._log_failure_to_supabase")
    def test_fallback_triggers_on_primary_failure(self, mock_log, mock_call):
        # Primary fails, fallback succeeds
        mock_call.side_effect = [
            Exception("primary down"),
            {
                "choices": [{"message": {"content": "fallback response"}}],
                "usage": {"total_tokens": 100},
            },
        ]

        config = ModelConfig(
            model="primary-model",
            provider="openrouter",
            effort="low",
            extended_thinking=False,
            fallback_model="fallback-model",
        )

        response = call(config, [{"role": "user", "content": "test"}])
        self.assertEqual(response.model_used, "fallback-model")
        self.assertTrue(response.fallback_triggered)
        self.assertEqual(response.content, "fallback response")
        mock_log.assert_called_once()

    @patch("src.agents.llm_client._call_openrouter")
    @patch("src.agents.llm_client._log_failure_to_supabase")
    def test_raises_when_both_fail(self, mock_log, mock_call):
        mock_call.side_effect = Exception("all down")

        config = ModelConfig(
            model="primary-model",
            provider="openrouter",
            effort="low",
            extended_thinking=False,
            fallback_model="fallback-model",
        )

        with self.assertRaises(LLMClientError):
            call(config, [{"role": "user", "content": "test"}])
        self.assertEqual(mock_log.call_count, 2)

    @patch("src.agents.llm_client._call_openrouter")
    def test_raises_when_no_fallback_configured(self, mock_call):
        mock_call.side_effect = Exception("down")

        config = ModelConfig(
            model="primary-model",
            provider="openrouter",
            effort="low",
            extended_thinking=False,
            fallback_model=None,
        )

        with self.assertRaises(LLMClientError):
            call(config, [{"role": "user", "content": "test"}])

    @patch("src.agents.llm_client._call_openrouter")
    def test_primary_success_no_fallback(self, mock_call):
        mock_call.return_value = {
            "choices": [{"message": {"content": "primary response"}}],
            "usage": {"total_tokens": 50},
        }

        config = get_performance_model()
        response = call(config, [{"role": "user", "content": "test"}])
        self.assertEqual(response.model_used, "moonshotai/kimi-k2.5")
        self.assertFalse(response.fallback_triggered)
        mock_call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
