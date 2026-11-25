"""Contract, profile-scoping, and lifecycle tests for tui_gateway native PDF attachments."""
from __future__ import annotations

import base64
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import owner_forward, server
from tui_gateway.contracts.prompt_voice import PdfAttachResult
from tui_gateway.session_auto_continue import _enqueue_prompt
from tui_gateway.session_history import _coerce_message_text

_TINY_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"
_TINY_PDF_B64 = base64.b64encode(_TINY_PDF).decode("ascii")


def _make_session(home: Path, sid: str, *, provider: str, model: str) -> dict:
    return {
        "agent": SimpleNamespace(provider=provider, model=model, base_url="", requested_provider=provider),
        "agent_ready": threading.Event(),
        "agent_error": None,
        "attached_images": [],
        "attached_documents": [],
        "cwd": str(home),
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "image_counter": 0,
        "profile_home": str(home),
        "running": False,
        "session_key": sid,
        "transport": None,
    }


def test_pdf_attach_rpc_contract_and_profile_scoped_routing(monkeypatch, tmp_path: Path):
    launch_home = tmp_path / "launch"
    secondary_home = tmp_path / "secondary"
    launch_home.mkdir()
    secondary_home.mkdir()
    (launch_home / "config.yaml").write_text("agent:\n  pdf_input_mode: text\n", encoding="utf-8")
    (secondary_home / "config.yaml").write_text("agent:\n  pdf_input_mode: native\n", encoding="utf-8")

    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr("shutil.which", lambda _name: None)

    sid = "pdf-native-profile-test"
    session = _make_session(secondary_home, sid, provider="solstice", model="gemini-4-neon")
    server._sessions[sid] = session
    try:
        # Launch profile sets pdf_input_mode=text, while session's profile_home sets pdf_input_mode=native.
        # @_profile_scoped must resolve secondary_home's config so native attach succeeds without pdftoppm.
        resp = server.handle_request({
            "id": "1",
            "method": "pdf.attach",
            "params": {"session_id": sid, "content_base64": _TINY_PDF_B64, "filename": "quarterly.pdf"},
        })
        assert "result" in resp, resp
        validated = PdfAttachResult.model_validate(resp["result"])
        assert validated.attached is True
        assert validated.native_document is True
        assert validated.path == session["attached_documents"][0]["path"]
        assert validated.filename == "quarterly.pdf"
        assert validated.pages_attached == 0
        assert validated.count == 1
        assert len(session["attached_documents"]) == 1
        assert Path(validated.path).name.startswith("quarterly")

        # image.detach removes the staged PDF using the path returned by pdf.attach and reports combined count=0
        detach_resp = server.handle_request({
            "id": "2",
            "method": "image.detach",
            "params": {"session_id": sid, "path": validated.path},
        })
        assert detach_resp["result"] == {"detached": True, "count": 0}
        assert session["attached_documents"] == []

        # Anthropic & Gemini 2.5 user turns also support native pdf.attach without pdftoppm
        for extra_sid, prov, mod, mode in (
            ("pdf-anthropic-test", "anthropic", "claude-sonnet-4-5", "anthropic_messages"),
            ("pdf-gemini25-test", "gemini", "gemini-2.5-pro", "chat_completions"),
        ):
            s_extra = _make_session(secondary_home, extra_sid, provider=prov, model=mod)
            s_extra["agent"].api_mode = mode
            server._sessions[extra_sid] = s_extra
            try:
                r_extra = server.handle_request({
                    "id": extra_sid,
                    "method": "pdf.attach",
                    "params": {"session_id": extra_sid, "content_base64": _TINY_PDF_B64, "filename": "brief.pdf"},
                })
                assert "result" in r_extra, r_extra
                v_extra = PdfAttachResult.model_validate(r_extra["result"])
                assert v_extra.native_document is True and v_extra.path
            finally:
                server._sessions.pop(extra_sid, None)
    finally:
        server._sessions.pop(sid, None)


def test_attached_documents_lifecycle_owner_forward_unstage_and_queue(tmp_path: Path):
    session = _make_session(tmp_path, "s-life", provider="solstice", model="gemini-4-neon")
    session["attached_documents"] = [{"path": str(tmp_path / "a.pdf"), "filename": "a.pdf", "mime_type": "application/pdf"}]

    refusal = SimpleNamespace(ledger_status="running")
    assert owner_forward.forwardable(refusal, session, "summarize", {}) is False

    # _enqueue_prompt does not merge doc-bearing turns into text-only queued_prompt
    _enqueue_prompt(session, "first", None)
    _enqueue_prompt(session, "second with doc", None, attached_documents=list(session["attached_documents"]))
    assert session["queued_prompt"]["text"] == "first"
    assert len(session["queued_prompts"]) == 1
    assert session["queued_prompts"][0]["attached_documents"][0]["filename"] == "a.pdf"

    # _unstage_refused_images clears both attached_images and attached_documents
    session["attached_images"] = [str(tmp_path / "img.png")]
    dropped = server._unstage_refused_images(session)
    assert dropped == 2
    assert session["attached_images"] == []
    assert session["attached_documents"] == []


def test_route_turn_documents_and_history_projection(tmp_path: Path):
    pdf_path = tmp_path / "deck.pdf"
    pdf_path.write_bytes(_TINY_PDF)
    spoof_path = tmp_path / "spoof.pdf"
    spoof_path.write_text("<html>fake</html>", encoding="utf-8")

    solstice_agent = SimpleNamespace(provider="solstice", model="gemini-4-neon", base_url="", requested_provider="solstice")
    deepseek_agent = SimpleNamespace(provider="deepseek", model="deepseek-chat", base_url="", requested_provider="deepseek")

    with patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}):
        native_out = server._route_turn_documents(
            solstice_agent,
            "Summarize this",
            [{"path": str(pdf_path), "filename": "deck.pdf", "mime_type": "application/pdf"}],
        )
        assert isinstance(native_out, list)
        assert native_out[0] == {"type": "text", "text": "Summarize this"}
        assert native_out[1]["type"] == "document" and native_out[1]["filename"] == "deck.pdf"

        # Spoofed PDF bytes degrade to path note rather than sending HTML as application/pdf
        spoof_out = server._route_turn_documents(
            solstice_agent,
            "Summarize this",
            [{"path": str(spoof_path), "filename": "spoof.pdf", "mime_type": "application/pdf"}],
        )
        assert isinstance(spoof_out, str) and "spoof.pdf" in spoof_out

        fb_out = server._route_turn_documents(
            deepseek_agent,
            "Summarize this",
            [{"path": str(pdf_path), "filename": "deck.pdf", "mime_type": "application/pdf"}],
        )
        assert isinstance(fb_out, str) and "deck.pdf" in fb_out

    big_b64 = "A" * 400_000
    rendered = _coerce_message_text(
        [{"type": "document", "filename": "a.pdf", "text": big_b64, "data": big_b64}],
        image_urls=False,
    )
    assert rendered.strip() == "[document: a.pdf]"


def test_queued_turn_attachment_isolation_in_admit_and_compute_host(tmp_path: Path, monkeypatch):
    session = _make_session(tmp_path, "s-iso", provider="solstice", model="gemini-4-neon")
    session["agent"].clear_interrupt = lambda: None
    live_img = str(tmp_path / "live.png")
    live_doc = {"path": str(tmp_path / "live.pdf"), "filename": "live.pdf", "mime_type": "application/pdf"}
    queued_doc = {"path": str(tmp_path / "queued.pdf"), "filename": "queued.pdf", "mime_type": "application/pdf"}
    session["attached_images"] = [live_img]
    session["attached_documents"] = [live_doc]

    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda _sid, _sess: None)

    # 1. Queued PDF-only turn must NOT steal live composer images in _admit_prompt_turn
    admitted = server._admit_prompt_turn(
        "s-iso", session, "queued pdf", None, None, None, None, attached_documents=[queued_doc],
    )
    assert admitted is not None
    images, _agent = admitted
    assert images == []
    assert session["_turn_attached_documents"] == [queued_doc]
    assert session["attached_images"] == [live_img]
    assert session["attached_documents"] == [live_doc]

    # 2. Compute-host turn frame: queued PDF-only does not steal live images, queued image-only does not steal live docs
    frame_pdf = server._compute_host_turn_frame(
        "r1", "s-iso", session, "queued pdf", image_paths=None, attached_documents=[queued_doc],
    )
    assert frame_pdf["attached_images"] == []
    assert frame_pdf["attached_documents"] == [queued_doc]

    frame_img = server._compute_host_turn_frame(
        "r2", "s-iso", session, "queued img", image_paths=[str(tmp_path / "q.png")], attached_documents=None,
    )
    assert frame_img["attached_images"] == [str(tmp_path / "q.png")]
    assert frame_img["attached_documents"] == []

    # 3. agent=None refusal must neither wipe live composer docs nor leave stale _turn_attached_documents
    session["agent"] = None
    monkeypatch.setattr(server, "_emit_terminal_turn_error", lambda *a, **k: None)
    assert server._admit_prompt_turn("s-iso", session, "hi", None, None, None, None) is None
    assert "_turn_attached_documents" not in session
    assert session["attached_documents"] == [live_doc]
    assert session["attached_images"] == [live_img]
