"""Shared bounded, symlink-safe media file I/O and MIME/magic validation helpers."""
from __future__ import annotations

import os
import stat
from typing import Optional

from tools.audio_container import sniff_container

MAX_NATIVE_MEDIA_RAW_BYTES = 15 * 1024 * 1024  # ~20 MB base64 inline cap

_AUDIO_MIME_ALIASES: dict[str, str] = {
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-pn-wav": "audio/wav",
    "audio/x-flac": "audio/flac",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/x-mp4": "audio/mp4",
    "audio/x-aiff": "audio/aiff",
    "audio/x-mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-ogg": "audio/ogg",
    "audio/x-aac": "audio/aac",
}

_TEXT_SPOOF_PREFIXES = (
    b"<!doctype",
    b"<html",
    b"<head",
    b"<body",
    b"<script",
    b"<?xml",
    b"<svg",
    b"{",
    b"[",
)


def normalize_audio_mime(mime: str) -> str:
    """Normalize non-canonical ``audio/x-*`` MIME aliases to canonical Gemini/Interactions types."""
    low = (mime or "").split(";", 1)[0].strip().lower()
    return _AUDIO_MIME_ALIASES.get(low, low)


def peek_file_header_nofollow(
    path: str,
    *,
    max_file_bytes: int = MAX_NATIVE_MEDIA_RAW_BYTES,
    header_bytes: int = 64,
) -> Optional[bytes]:
    """Validate a regular local file via ``O_NOFOLLOW`` + ``fstat`` (``0 < size <= max_file_bytes``)
    and return only its first *header_bytes* without allocating the full file in RAM."""
    if not path or max_file_bytes <= 0 or header_bytes <= 0:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size <= 0 or st.st_size > max_file_bytes:
            return None
        fp = os.fdopen(fd, "rb")
        fd = -1
        with fp:
            head = fp.read(header_bytes)
        return head if head else None
    except OSError:
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def read_file_bounded_nofollow(path: str, *, max_bytes: int = MAX_NATIVE_MEDIA_RAW_BYTES) -> Optional[bytes]:
    """Read a regular local file without following symlinks and without loading oversized files into RAM.

    Returns ``None`` on symlink (``ELOOP``), non-regular file, empty/oversized file, or OS error.
    Strictly closes ``fd`` on every error path without double-closing after ``os.fdopen`` handoff.
    """
    if not path or max_bytes <= 0:
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size <= 0 or st.st_size > max_bytes:
            return None
        fp = os.fdopen(fd, "rb")
        fd = -1  # ownership transferred to fp; finally will not double-close
        with fp:
            raw = bytearray()
            while len(raw) <= max_bytes:
                chunk = fp.read(min(65536, max_bytes - len(raw) + 1))
                if not chunk:
                    break
                raw.extend(chunk)
        return bytes(raw) if 0 < len(raw) <= max_bytes else None
    except OSError:
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def is_valid_pdf_bytes(data: bytes) -> bool:
    """True when *data* begins with a genuine ``%PDF-`` header."""
    return bool(data) and data[:5] == b"%PDF-"


def is_valid_audio_bytes(data: bytes) -> bool:
    """True when *data* looks like a genuine audio container (not HTML/XML/JSON/BOM text)."""
    if not data or len(data) < 4:
        return False
    # UTF-16LE/BE BOMs (0xFF 0xFE / 0xFE 0xFF) would otherwise trip the 0xFF 0xE0 MP3 sync check.
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return False
    head = data[:64].lstrip().lower()
    if any(head.startswith(prefix) for prefix in _TEXT_SPOOF_PREFIXES):
        return False
    # Reject raw PDF disguised as audio
    if data[:5] == b"%PDF-":
        return False
    if len(data) >= 12 and data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return True
    return sniff_container(data[:64]) is not None
