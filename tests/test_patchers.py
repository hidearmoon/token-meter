"""Tests for SDK monkey-patchers using mock objects."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import pytest
from token_meter.config import TokenMeterConfig
from token_meter.models import UsageRecord
from token_meter.storage.sqlite import SQLiteStorage
from token_meter.patchers.openai import OpenAIPatcher
from token_meter.patchers.anthropic import AnthropicPatcher
from token_meter.patchers.google import GooglePatcher


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    storage = SQLiteStorage(tmp_path / "test.db")
    yield storage
    storage.close()


@pytest.fixture
def config():
    return TokenMeterConfig(project="test-project")


# ── OpenAI Patcher ────────────────────────────────────────────────────────────

def _openai_response(model="gpt-4o", in_tokens=100, out_tokens=50):
    usage = SimpleNamespace(
        prompt_tokens=in_tokens,
        completion_tokens=out_tokens,
    )
    return SimpleNamespace(model=model, usage=usage)


class TestOpenAIPatcher:
    def _make_patcher(self, config, storage):
        return OpenAIPatcher(config, storage)

    def test_patch_and_unpatch_restores_original(self, config, tmp_db):
        """Patching and unpatching should leave the original method in place."""
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        original_sync = _mod.Completions.create
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        assert _mod.Completions.create is not original_sync

        patcher.unpatch()
        assert _mod.Completions.create is original_sync

    def test_non_stream_records_usage(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        response = _openai_response("gpt-4o", 200, 80)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            # Call the patched create
            with patch.object(_mod.Completions, "create", wraps=_mod.Completions.create):
                # Simulate the patched wrapper calling _original
                patcher._original = MagicMock(return_value=response)
                _mod.Completions.create(mock_client, model="gpt-4o", messages=[])
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.provider == "openai"
        assert r.model == "gpt-4o"
        assert r.input_tokens == 200
        assert r.output_tokens == 80
        assert r.is_stream is False
        assert r.project == "test-project"

    def test_non_stream_cost_calculated(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        response = _openai_response("gpt-4o-mini", 1_000_000, 1_000_000)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original = MagicMock(return_value=response)
            _mod.Completions.create(mock_client, model="gpt-4o-mini", messages=[])
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert records[0].input_cost == pytest.approx(0.15, rel=1e-3)
        assert records[0].output_cost == pytest.approx(0.60, rel=1e-3)

    def test_double_patch_is_idempotent(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        first_fn = _mod.Completions.create
        patcher.patch()  # second call
        assert _mod.Completions.create is first_fn
        patcher.unpatch()

    def test_unpatch_without_patch_is_noop(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        patcher = self._make_patcher(config, tmp_db)
        original = _mod.Completions.create
        patcher.unpatch()  # should not raise
        assert _mod.Completions.create is original

    def test_stream_injects_include_usage(self, config, tmp_db):
        """Verify that stream_options={'include_usage': True} is injected."""
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        captured_kwargs = {}

        def fake_original(self_client, *args, **kwargs):
            captured_kwargs.update(kwargs)
            # Return empty iterator
            return iter([])

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original = fake_original
        try:
            mock_client = MagicMock()
            gen = _mod.Completions.create(mock_client, model="gpt-4o", messages=[], stream=True)
            list(gen)  # exhaust
        finally:
            patcher.unpatch()

        assert captured_kwargs.get("stream_options", {}).get("include_usage") is True

    def test_stream_preserves_existing_stream_options(self, config, tmp_db):
        """Existing stream_options keys should not be overwritten."""
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        captured_kwargs = {}

        def fake_original(self_client, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return iter([])

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original = fake_original
        try:
            mock_client = MagicMock()
            gen = _mod.Completions.create(
                mock_client,
                model="gpt-4o",
                messages=[],
                stream=True,
                stream_options={"other_key": "other_val"},
            )
            list(gen)
        finally:
            patcher.unpatch()

        so = captured_kwargs.get("stream_options", {})
        assert so.get("include_usage") is True
        assert so.get("other_key") == "other_val"

    def test_stream_records_usage_from_last_chunk(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        usage_obj = SimpleNamespace(prompt_tokens=300, completion_tokens=120)
        chunks = [
            SimpleNamespace(model="gpt-4o", usage=None),
            SimpleNamespace(model="gpt-4o", usage=None),
            SimpleNamespace(model="gpt-4o", usage=usage_obj),
        ]

        def fake_original(self_client, *args, **kwargs):
            return iter(chunks)

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original = fake_original
        try:
            mock_client = MagicMock()
            gen = _mod.Completions.create(mock_client, model="gpt-4o", messages=[], stream=True)
            list(gen)
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        assert records[0].input_tokens == 300
        assert records[0].output_tokens == 120
        assert records[0].is_stream is True


# ── Anthropic Patcher ─────────────────────────────────────────────────────────

def _anthropic_response(model="claude-sonnet-4", in_tokens=100, out_tokens=50):
    usage = SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens)
    return SimpleNamespace(model=model, usage=usage)


class TestAnthropicPatcher:
    def _make_patcher(self, config, storage):
        return AnthropicPatcher(config, storage)

    def test_patch_and_unpatch(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        original = _mod.Messages.create
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        assert _mod.Messages.create is not original
        patcher.unpatch()
        assert _mod.Messages.create is original

    def test_non_stream_records_usage(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        response = _anthropic_response("claude-sonnet-4", 150, 75)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original = MagicMock(return_value=response)
            _mod.Messages.create(mock_client, model="claude-sonnet-4", messages=[])
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.provider == "anthropic"
        assert r.model == "claude-sonnet-4"
        assert r.input_tokens == 150
        assert r.output_tokens == 75

    def test_stream_accumulates_tokens_from_events(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        # Simulate SSE event objects
        start_usage = SimpleNamespace(input_tokens=200)
        start_msg = SimpleNamespace(model="claude-opus-4", usage=start_usage)
        msg_start = SimpleNamespace(type="message_start", message=start_msg)

        delta_usage = SimpleNamespace(output_tokens=80)
        msg_delta = SimpleNamespace(type="message_delta", usage=delta_usage)

        events = [msg_start, msg_delta]

        def fake_original(self_client, *args, **kwargs):
            return iter(events)

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original = fake_original
        try:
            mock_client = MagicMock()
            gen = _mod.Messages.create(mock_client, model="claude-opus-4", messages=[], stream=True)
            list(gen)
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.input_tokens == 200
        assert r.output_tokens == 80
        assert r.is_stream is True
        assert r.model == "claude-opus-4"


# ── Google Patcher ────────────────────────────────────────────────────────────

def _google_response(model="gemini-2.5-flash", in_tokens=100, out_tokens=50):
    meta = SimpleNamespace(
        prompt_token_count=in_tokens,
        candidates_token_count=out_tokens,
        total_token_count=in_tokens + out_tokens,
    )
    return SimpleNamespace(model=model, usage_metadata=meta)


class TestGooglePatcher:
    def _make_patcher(self, config, storage):
        return GooglePatcher(config, storage)

    def test_patch_and_unpatch(self, config, tmp_db):
        try:
            import google.genai.models as _mod
        except ImportError:
            pytest.skip("google-genai not installed")

        original = _mod.Models.generate_content
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        assert _mod.Models.generate_content is not original
        patcher.unpatch()
        assert _mod.Models.generate_content is original

    def test_non_stream_records_usage(self, config, tmp_db):
        try:
            import google.genai.models as _mod
        except ImportError:
            pytest.skip("google-genai not installed")

        response = _google_response("gemini-2.5-pro", 500, 200)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original = MagicMock(return_value=response)
            _mod.Models.generate_content(
                mock_client, model="gemini-2.5-pro", contents="Hello"
            )
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.provider == "google"
        assert r.input_tokens == 500
        assert r.output_tokens == 200

    def test_stream_captures_last_chunk_usage(self, config, tmp_db):
        try:
            import google.genai.models as _mod
        except ImportError:
            pytest.skip("google-genai not installed")

        # Stream chunks — only last has complete usage_metadata
        chunk1 = SimpleNamespace(usage_metadata=None)
        chunk2 = SimpleNamespace(usage_metadata=None)
        meta = SimpleNamespace(
            prompt_token_count=400,
            candidates_token_count=150,
            total_token_count=550,
        )
        chunk3 = SimpleNamespace(usage_metadata=meta)

        def fake_original(self_client, *args, **kwargs):
            return iter([chunk1, chunk2, chunk3])

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original = fake_original
        try:
            mock_client = MagicMock()
            gen = _mod.Models.generate_content(
                mock_client, model="gemini-2.5-flash", contents="Hi", stream=True
            )
            list(gen)
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.input_tokens == 400
        assert r.output_tokens == 150
        assert r.total_tokens == 550
        assert r.is_stream is True

    def test_models_prefix_stripped(self, config, tmp_db):
        """'models/gemini-2.5-flash' should be stored as 'gemini-2.5-flash'."""
        try:
            import google.genai.models as _mod
        except ImportError:
            pytest.skip("google-genai not installed")

        response = _google_response("gemini-2.5-flash", 10, 5)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        try:
            mock_client = MagicMock()
            patcher._original = MagicMock(return_value=response)
            _mod.Models.generate_content(
                mock_client, model="models/gemini-2.5-flash", contents="Hi"
            )
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert records[0].model == "gemini-2.5-flash"


# ── Async Patcher Tests ───────────────────────────────────────────────────────

class TestOpenAIPatcherAsync:
    def _make_patcher(self, config, storage):
        return OpenAIPatcher(config, storage)

    async def test_async_non_stream_records_usage(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        response = _openai_response("gpt-4o", 200, 80)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original_async = AsyncMock(return_value=response)
            result = await _mod.AsyncCompletions.create(
                mock_client, model="gpt-4o", messages=[]
            )
            assert result is response
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.provider == "openai"
        assert r.input_tokens == 200
        assert r.output_tokens == 80
        assert r.is_stream is False

    async def test_async_non_stream_no_usage_skipped(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        response = SimpleNamespace(model="gpt-4o", usage=None)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original_async = AsyncMock(return_value=response)
            await _mod.AsyncCompletions.create(mock_client, model="gpt-4o", messages=[])
        finally:
            patcher.unpatch()

        assert tmp_db.query() == []

    async def test_async_stream_records_usage(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        usage_obj = SimpleNamespace(prompt_tokens=300, completion_tokens=120)
        chunks = [
            SimpleNamespace(model="gpt-4o", usage=None),
            SimpleNamespace(model="gpt-4o", usage=usage_obj),
        ]

        async def fake_stream():
            for chunk in chunks:
                yield chunk

        async def fake_original_async(self_client, *args, **kwargs):
            return fake_stream()

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original_async = fake_original_async

        try:
            mock_client = MagicMock()
            gen = await _mod.AsyncCompletions.create(
                mock_client, model="gpt-4o", messages=[], stream=True
            )
            async for _ in gen:
                pass
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        assert records[0].input_tokens == 300
        assert records[0].output_tokens == 120
        assert records[0].is_stream is True

    async def test_async_stream_injects_include_usage(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        captured_kwargs: dict = {}

        async def fake_original_async(self_client, *args, **kwargs):
            captured_kwargs.update(kwargs)

            async def empty():
                return
                yield  # make it an async generator

            return empty()

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original_async = fake_original_async

        try:
            mock_client = MagicMock()
            gen = await _mod.AsyncCompletions.create(
                mock_client, model="gpt-4o", messages=[], stream=True
            )
            async for _ in gen:
                pass
        finally:
            patcher.unpatch()

        assert captured_kwargs.get("stream_options", {}).get("include_usage") is True


class TestAnthropicPatcherAsync:
    def _make_patcher(self, config, storage):
        return AnthropicPatcher(config, storage)

    async def test_async_non_stream_records_usage(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        response = _anthropic_response("claude-sonnet-4", 150, 75)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original_async = AsyncMock(return_value=response)
            result = await _mod.AsyncMessages.create(
                mock_client, model="claude-sonnet-4", messages=[]
            )
            assert result is response
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.provider == "anthropic"
        assert r.input_tokens == 150
        assert r.output_tokens == 75
        assert r.is_stream is False

    async def test_async_non_stream_no_usage_skipped(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        response = SimpleNamespace(model="claude-sonnet-4", usage=None)
        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()

        try:
            mock_client = MagicMock()
            patcher._original_async = AsyncMock(return_value=response)
            await _mod.AsyncMessages.create(mock_client, model="claude-sonnet-4", messages=[])
        finally:
            patcher.unpatch()

        assert tmp_db.query() == []

    async def test_async_stream_accumulates_tokens(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        start_usage = SimpleNamespace(input_tokens=200)
        start_msg = SimpleNamespace(model="claude-opus-4", usage=start_usage)
        msg_start = SimpleNamespace(type="message_start", message=start_msg)
        delta_usage = SimpleNamespace(output_tokens=80)
        msg_delta = SimpleNamespace(type="message_delta", usage=delta_usage)
        events = [msg_start, msg_delta]

        async def fake_stream():
            for event in events:
                yield event

        async def fake_original_async(self_client, *args, **kwargs):
            return fake_stream()

        patcher = self._make_patcher(config, tmp_db)
        patcher.patch()
        patcher._original_async = fake_original_async

        try:
            mock_client = MagicMock()
            gen = await _mod.AsyncMessages.create(
                mock_client, model="claude-opus-4", messages=[], stream=True
            )
            async for _ in gen:
                pass
        finally:
            patcher.unpatch()

        records = tmp_db.query()
        assert len(records) == 1
        r = records[0]
        assert r.input_tokens == 200
        assert r.output_tokens == 80
        assert r.is_stream is True
        assert r.model == "claude-opus-4"


# ── Base Patcher edge cases ───────────────────────────────────────────────────

class TestBasePatcherEdgeCases:
    def test_patch_unexpected_exception_returns_false(self, config, tmp_db):
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai not installed")

        patcher = OpenAIPatcher(config, tmp_db)

        def bad_patch():
            raise RuntimeError("unexpected error during patch")

        patcher._do_patch = bad_patch
        result = patcher.patch()
        assert result is False
        assert not patcher.is_active

    def test_unpatch_exception_handled_gracefully(self, config, tmp_db):
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai not installed")

        patcher = OpenAIPatcher(config, tmp_db)
        patcher.patch()
        assert patcher.is_active

        def bad_unpatch():
            raise RuntimeError("unpatch failed")

        patcher._do_unpatch = bad_unpatch
        patcher.unpatch()  # must not raise
        assert not patcher.is_active  # is_active reset in finally block

    def test_non_stream_no_usage_not_recorded_openai(self, config, tmp_db):
        try:
            import openai.resources.chat.completions as _mod
        except ImportError:
            pytest.skip("openai not installed")

        response = SimpleNamespace(model="gpt-4o", usage=None)
        patcher = OpenAIPatcher(config, tmp_db)
        patcher.patch()
        try:
            patcher._original = MagicMock(return_value=response)
            _mod.Completions.create(MagicMock(), model="gpt-4o", messages=[])
        finally:
            patcher.unpatch()
        assert tmp_db.query() == []

    def test_non_stream_no_usage_not_recorded_anthropic(self, config, tmp_db):
        try:
            import anthropic.resources.messages as _mod
        except ImportError:
            pytest.skip("anthropic not installed")

        response = SimpleNamespace(model="claude-sonnet-4", usage=None)
        patcher = AnthropicPatcher(config, tmp_db)
        patcher.patch()
        try:
            patcher._original = MagicMock(return_value=response)
            _mod.Messages.create(MagicMock(), model="claude-sonnet-4", messages=[])
        finally:
            patcher.unpatch()
        assert tmp_db.query() == []
