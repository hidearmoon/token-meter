"""Tests for the pricing engine."""
import pytest
from token_meter.pricing import get_cost, list_models, add_custom_pricing, _match_model


class TestGetCost:
    def test_known_model_gpt4o(self):
        in_cost, out_cost, total = get_cost("gpt-4o", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(2.50, rel=1e-5)
        assert out_cost == pytest.approx(10.00, rel=1e-5)
        assert total == pytest.approx(12.50, rel=1e-5)

    def test_known_model_gpt4o_mini(self):
        in_cost, out_cost, total = get_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(0.15, rel=1e-5)
        assert out_cost == pytest.approx(0.60, rel=1e-5)

    def test_known_model_claude_sonnet4(self):
        in_cost, out_cost, total = get_cost("claude-sonnet-4", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(3.00, rel=1e-5)
        assert out_cost == pytest.approx(15.00, rel=1e-5)

    def test_known_model_claude_opus4(self):
        in_cost, out_cost, total = get_cost("claude-opus-4", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(15.00, rel=1e-5)
        assert out_cost == pytest.approx(75.00, rel=1e-5)

    def test_known_model_gemini_25_pro(self):
        in_cost, out_cost, total = get_cost("gemini-2.5-pro", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(1.25, rel=1e-5)
        assert out_cost == pytest.approx(10.00, rel=1e-5)

    def test_known_model_gemini_25_flash(self):
        in_cost, out_cost, total = get_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(0.30, rel=1e-5)
        assert out_cost == pytest.approx(2.50, rel=1e-5)

    def test_small_token_count(self):
        # 1000 input tokens with gpt-4o: 1000 * 2.50 / 1_000_000 = 0.0025
        in_cost, out_cost, total = get_cost("gpt-4o", 1000, 500)
        assert in_cost == pytest.approx(0.0025, rel=1e-4)
        assert out_cost == pytest.approx(0.005, rel=1e-4)

    def test_zero_tokens(self):
        in_cost, out_cost, total = get_cost("gpt-4o", 0, 0)
        assert in_cost == 0.0
        assert out_cost == 0.0
        assert total == 0.0

    def test_unknown_model_returns_zero(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            in_cost, out_cost, total = get_cost(
                "nonexistent-model-xyz", 1000, 1000, warn_unknown=True
            )
        assert in_cost == 0.0
        assert out_cost == 0.0
        assert total == 0.0
        assert "unknown model" in caplog.text

    def test_unknown_model_no_warn(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            in_cost, _, _ = get_cost("nope", 100, 100, warn_unknown=False)
        assert in_cost == 0.0
        assert "unknown model" not in caplog.text

    def test_cost_precision_6_decimals(self):
        # 1 token with gpt-4o-mini input = 0.15 / 1_000_000 = 0.00000015
        in_cost, _, _ = get_cost("gpt-4o-mini", 1, 0)
        assert len(str(in_cost).split(".")[-1]) <= 6

    def test_total_equals_sum(self):
        in_cost, out_cost, total = get_cost("gpt-4o", 12345, 67890)
        assert total == pytest.approx(in_cost + out_cost, rel=1e-9)


class TestModelMatching:
    def test_exact_match(self):
        assert _match_model("gpt-4o") == "gpt-4o"

    def test_case_insensitive(self):
        assert _match_model("GPT-4O") == "gpt-4o"
        assert _match_model("Claude-Sonnet-4") == "claude-sonnet-4"

    def test_date_suffix_openai(self):
        assert _match_model("gpt-4o-2024-11-20") == "gpt-4o"

    def test_date_suffix_anthropic(self):
        assert _match_model("claude-sonnet-4-20250514") == "claude-sonnet-4"

    def test_preview_suffix(self):
        assert _match_model("gpt-4o-preview") == "gpt-4o"

    def test_unknown_returns_none(self):
        assert _match_model("llama-99") is None

    def test_fuzzy_prefix_match(self):
        # "gpt-4o-mini-2024-07-18" should fuzzy match gpt-4o-mini
        assert _match_model("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"


class TestCustomPricing:
    def test_add_and_use_custom_model(self):
        add_custom_pricing("my-finetune-v1", 5.00, 20.00)
        in_cost, out_cost, total = get_cost("my-finetune-v1", 1_000_000, 1_000_000)
        assert in_cost == pytest.approx(5.00)
        assert out_cost == pytest.approx(20.00)

    def test_list_models_returns_dict(self):
        models = list_models()
        assert isinstance(models, dict)
        assert "gpt-4o" in models
        assert "claude-sonnet-4" in models
        assert "gemini-2.5-pro" in models
