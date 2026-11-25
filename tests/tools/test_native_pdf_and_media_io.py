"""Contract & security tests for read_file native PDF fast-path, tools.media_io, and OOXML zip-bomb bounds."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
from tools import file_tools as _file_tools  # registers read_file on registry
from tools.media_io import is_valid_audio_bytes, is_valid_pdf_bytes, normalize_audio_mime, read_file_bounded_nofollow
from tools.file_tools_read_tracking import _programmatic_read
from tools.read_extract import ExtractionError, _extract_docx, _extract_xlsx
from tools.registry import registry
from tools.vision_tools_history_budget import _repeat_counts

_TINY_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"


def test_read_file_registry_dispatch_execute_code_and_subagent_cap(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_TINY_PDF)

    _repeat_counts.clear()
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    with (
        patch("agent.image_routing._runtime_main", side_effect=lambda k: {"provider": "gemini", "model": "gemini-4-neon", "base_url": "", "requested_provider": "gemini"}.get(k, "")),
        patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}),
    ):
        # 1. Interactive model dispatch via registry -> native _multimodal envelope
        res = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
        assert isinstance(res, dict) and res.get("_multimodal") is True
        assert res["content"][1]["type"] == "document"
        assert res["content"][1]["mime_type"] == "application/pdf"

        # 2. execute_code (_programmatic_read=True) -> NEVER returns _multimodal dict
        tok = _programmatic_read.set(True)
        try:
            prog_res = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
            assert isinstance(prog_res, str)
        finally:
            _programmatic_read.reset(tok)

        # 3. Explicit pagination (offset > 1) -> stays on text extraction
        paged_res = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 2, "limit": 100})
        assert isinstance(paged_res, str)

        # 4. Subagent reserve_embed cap (3): 4th read is refused with JSON error, not _multimodal
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        monkeypatch.setenv("HERMES_SESSION_ID", f"subagent-{tmp_path.name}")
        for _ in range(3):
            ok = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
            assert isinstance(ok, dict) and ok.get("_multimodal") is True
        refused = registry.dispatch("read_file", {"path": str(pdf_path), "offset": 1, "limit": 100})
        assert isinstance(refused, str)
        parsed_refusal = json.loads(refused)
        assert parsed_refusal.get("success") is False
        assert "3 time(s)" in parsed_refusal.get("error", "")


def test_read_file_rejects_spoofed_pdf_magic_and_media_io_security(tmp_path: Path):
    fake_pdf = tmp_path / "spoofed.pdf"
    fake_pdf.write_text("<html>not a pdf</html>", encoding="utf-8")
    with (
        patch("agent.image_routing._runtime_main", side_effect=lambda k: {"provider": "gemini", "model": "gemini-4-neon", "base_url": "", "requested_provider": "gemini"}.get(k, "")),
        patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}),
    ):
        res = registry.dispatch("read_file", {"path": str(fake_pdf), "offset": 1, "limit": 100})
        assert isinstance(res, str)

    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(_TINY_PDF)
    sym_pdf = tmp_path / "link.pdf"
    sym_pdf.symlink_to(real_pdf)
    assert read_file_bounded_nofollow(str(sym_pdf)) is None
    assert is_valid_pdf_bytes(_TINY_PDF) is True
    assert is_valid_pdf_bytes(b"<html>fake</html>") is False

    # Audio magic validation: reject HTML and UTF-16LE/BE BOMs; accept WAV & AIFF; normalize aliases
    assert is_valid_audio_bytes(b"<html><body>not audio</body></html>") is False
    assert is_valid_audio_bytes(b"\xff\xfeH\x00e\x00l\x00l\x00o\x00") is False
    assert is_valid_audio_bytes(b"\xfe\xff\x00H\x00e\x00l\x00l\x00o") is False
    assert is_valid_audio_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ") is True
    assert is_valid_audio_bytes(b"FORM\x00\x00\x00\x24AIFFCOMM") is True
    assert normalize_audio_mime("audio/x-wav") == "audio/wav"


def test_read_extract_docx_and_multisheet_xlsx_zip_bombs_rejected(tmp_path: Path):
    # Single-member DOCX zip bomb (> 25 MB member cap)
    docx_bomb = tmp_path / "bomb.docx"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"0" * (26 * 1024 * 1024))
    docx_bomb.write_bytes(buf.getvalue())
    with pytest.raises(ExtractionError, match="exceeds safe decompressed size limit"):
        _extract_docx(str(docx_bomb))

    # Multi-sheet cumulative XLSX zip bomb: each sheet is < 25 MB, total > 100 MB
    xlsx_bomb = tmp_path / "sheets_bomb.xlsx"
    xbuf = io.BytesIO()
    with zipfile.ZipFile(xbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        sheets_xml = "".join(
            f'<sheet name="S{i}" sheetId="{i}" r:id="rId{i}"/>' for i in range(1, 6)
        )
        rels_xml = "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, 6)
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{sheets_xml}</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{rels_xml}</Relationships>",
        )
        pad = b" " * (22 * 1024 * 1024)
        sheet_body = (
            b'<?xml version="1.0"?>'
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<sheetData><row><c><v>1</v></c></row></sheetData><!--"
            + pad
            + b"--></worksheet>"
        )
        for i in range(1, 6):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_body)
    xlsx_bomb.write_bytes(xbuf.getvalue())
    with pytest.raises(ExtractionError, match="exceeds safe decompressed size limit"):
        _extract_xlsx(str(xlsx_bomb))


def test_read_file_native_pdf_never_bypasses_secret_redaction(tmp_path: Path, monkeypatch):
    import zlib

    secret = "ghp_1234567890abcdefghij1234567890abcd"
    plain_secret_pdf = tmp_path / "plain_secret.pdf"
    plain_secret_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n"
        + f"% Token: {secret}\n".encode("ascii")
        + b"%%EOF\n"
    )
    flate_stream = zlib.compress(f"BT ({secret}) Tj ET".encode("ascii"))
    flate_secret_pdf = tmp_path / "flate_secret.pdf"
    flate_secret_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj << /Filter /FlateDecode >>\nstream\n"
        + flate_stream
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    tj_split_pdf = tmp_path / "tj_secret.pdf"
    tj_split_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj\nstream\n"
        + zlib.compress(f"BT [({secret[:14]}) -10 ({secret[14:]})] TJ ET".encode("ascii"))
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    clean_pdf = tmp_path / "clean.pdf"
    clean_pdf.write_bytes(_TINY_PDF)

    _repeat_counts.clear()
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    with (
        patch("agent.image_routing._runtime_main", side_effect=lambda k: {"provider": "gemini", "model": "gemini-4-neon", "base_url": "", "requested_provider": "gemini"}.get(k, "")),
        patch("hermes_cli.config.load_config", return_value={"agent": {"pdf_input_mode": "native"}}),
    ):
        for secret_pdf in (plain_secret_pdf, flate_secret_pdf, tj_split_pdf):
            res = registry.dispatch("read_file", {"path": str(secret_pdf), "offset": 1, "limit": 100})
            assert isinstance(res, str), f"Secret-bearing PDF {secret_pdf.name} must NOT attach natively"
            assert secret not in res

        clean_res = registry.dispatch("read_file", {"path": str(clean_pdf), "offset": 1, "limit": 100})
        assert isinstance(clean_res, dict) and clean_res.get("_multimodal") is True
