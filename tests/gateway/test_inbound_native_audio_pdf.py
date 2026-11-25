"""Contract tests for gateway inbound native audio & PDF routing and TurnRunner materialization."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.run_inbound import GatewayInboundMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session_state import PersistentState

_TINY_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"


def test_gateway_inbound_native_audio_pdf_auto_and_spoof_rejection(tmp_path: Path):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(_TINY_PDF)
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    spoof_audio = tmp_path / "fake.mp3"
    spoof_audio.write_text("<html>not audio</html>", encoding="utf-8")

    class _FakeRunner(GatewayInboundMixin):
        def __init__(self, provider: str, model: str):
            self._provider = provider
            self._model = model
            self._st = SimpleNamespace(persistent=PersistentState())

        def _resolve_session_agent_runtime(self, *, source, session_key):
            return self._model, {"provider": self._provider, "base_url": "", "requested_provider": self._provider}

        def _session_state(self, session_key):
            return self._st

        def _peek_session_state(self, session_key):
            return self._st

    event = SimpleNamespace(
        media_urls=[str(wav_path), str(pdf_path), str(spoof_audio)],
        media_types=["audio/x-wav", "application/pdf", "audio/mpeg"],
        media_filenames=["clip.wav", "invoice.pdf", "fake.mp3"],
    )
    # Default empty config {} activates native audio & PDF on solstice/gemini-4-neon
    with patch("hermes_cli.config.load_config", return_value={}):
        runner = _FakeRunner("solstice", "gemini-4-neon")
        rem_audio, native_pdfs = runner._route_inbound_native_audio_and_pdfs(
            event, None, "s1", [str(wav_path), str(spoof_audio)]
        )
        # Valid WAV is queued with normalized audio/wav; HTML-spoofed fake.mp3 stays on fallback
        assert rem_audio == [str(spoof_audio)]
        assert native_pdfs == {str(pdf_path)}
        queued = runner._consume_pending_native_media_attachments("s1")
        assert [(q["kind"], q["mime_type"]) for q in queued] == [("audio", "audio/wav"), ("document", "application/pdf")]
        assert runner._consume_pending_native_media_attachments("s1") == []

        # Non-native model leaves both audio_file_paths and PDFs on fallback path
        ds = _FakeRunner("deepseek", "deepseek-chat")
        rem_audio_ds, native_pdfs_ds = ds._route_inbound_native_audio_and_pdfs(
            event, None, "s2", [str(wav_path)]
        )
        assert rem_audio_ds == [str(wav_path)]
        assert native_pdfs_ds == set()


def test_turn_runner_native_media_materialization(tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_TINY_PDF)
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    fake_runner = SimpleNamespace(
        _consume_pending_native_image_paths=lambda sid: [],
        _consume_pending_native_media_attachments=lambda sid: [
            {"kind": "audio", "path": str(wav_path), "mime_type": "audio/x-wav", "filename": "voice.wav"},
            {"kind": "document", "path": str(pdf_path), "mime_type": "application/pdf", "filename": "doc.pdf"},
        ],
    )
    tr = object.__new__(TurnRunner)
    tr._runner = fake_runner
    tr._ctx = SimpleNamespace(session_key="s1", message="Check attachments")
    out = tr._native_image_run_message()
    assert isinstance(out, list) and len(out) == 3
    assert out[1]["type"] == "audio" and out[1]["mime_type"] == "audio/wav"
    assert out[2]["type"] == "document" and out[2]["filename"] == "doc.pdf"
