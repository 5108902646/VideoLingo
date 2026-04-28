import io
import os
import re
import threading
import time
from pathlib import Path

import requests
from pydub import AudioSegment

from core.utils import load_key, rprint

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TIME = 0.0


def _load_proxy_config():
    max_retries = int(load_key("proxy_tts.max_retries"))
    chunk_size = int(load_key("proxy_tts.chunk_size"))
    request_cooldown_ms = int(load_key("proxy_tts.request_cooldown_ms"))
    timeout_sec = int(load_key("proxy_tts.timeout_sec"))
    return {
        "endpoint_url": load_key("proxy_tts.endpoint_url").strip(),
        "voice": load_key("proxy_tts.voice"),
        "model": load_key("proxy_tts.model"),
        "speed": float(load_key("proxy_tts.speed")),
        "chunk_size": max(1, chunk_size),
        "request_cooldown_ms": max(0, request_cooldown_ms),
        "max_retries": max(1, max_retries),
        "timeout_sec": max(1, timeout_sec),
    }


def _is_cjk_character(char):
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
    )


def _tiktok_character_count(text):
    return sum(2 if _is_cjk_character(char) else 1 for char in text)


def _split_by_tiktok_count(text, max_count):
    chunks = []
    current_chunk = ""
    current_count = 0

    for char in text:
        char_count = 2 if _is_cjk_character(char) else 1
        if current_count + char_count > max_count:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = char
            current_count = char_count
        else:
            current_chunk += char
            current_count += char_count

    if current_chunk:
        chunks.append(current_chunk)
    return [chunk for chunk in chunks if chunk.strip()]


def _split_japanese_text(text, max_chars):
    chunks = []
    sentences = [part.strip() for part in re.split(r"(?<=[。！？])", text) if part.strip()]
    current_chunk = ""
    current_count = 0

    for sentence in sentences:
        sentence_count = _tiktok_character_count(sentence)
        if sentence_count > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_count = 0
            chunks.extend(_split_by_tiktok_count(sentence, max_chars))
        elif current_count + sentence_count <= max_chars:
            current_chunk += sentence
            current_count += sentence_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_count = sentence_count

    if current_chunk:
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _split_regular_text(text, max_chars):
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks = []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？ฯ])\s*", text) if part.strip()]
    current_chunk = ""

    for sentence in sentences:
        test = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        if len(test) <= max_chars:
            current_chunk = test
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(sentence) <= max_chars:
            current_chunk = sentence
            continue

        clauses = [part.strip() for part in re.split(r"(?<=[,;:،؛、，；：])\s*", sentence) if part.strip()]
        current_chunk = ""
        for clause in clauses:
            test_clause = f"{current_chunk} {clause}".strip() if current_chunk else clause
            if len(test_clause) <= max_chars:
                current_chunk = test_clause
                continue

            if current_chunk:
                chunks.append(current_chunk)

            if len(clause) <= max_chars:
                current_chunk = clause
                continue

            words = clause.split()
            if not words:
                sub_chunks = [clause[i : i + max_chars] for i in range(0, len(clause), max_chars)]
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] if sub_chunks else ""
                continue

            current_chunk = ""
            for word in words:
                test_word = f"{current_chunk} {word}".strip() if current_chunk else word
                if len(test_word) <= max_chars:
                    current_chunk = test_word
                elif len(word) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                    word_chunks = [word[i : i + max_chars] for i in range(0, len(word), max_chars)]
                    chunks.extend(word_chunks[:-1])
                    current_chunk = word_chunks[-1] if word_chunks else ""
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)
    return [chunk for chunk in chunks if chunk.strip()]


def split_text_into_chunks(text, max_chars=150):
    text = (text or "").strip()
    if not text:
        return []

    if re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text):
        if _tiktok_character_count(text) <= max_chars:
            return [text]
        return _split_japanese_text(text, max_chars)

    return _split_regular_text(text, max_chars)


def _wait_for_request_slot(cooldown_ms):
    global _LAST_REQUEST_TIME
    if cooldown_ms <= 0:
        return

    with _REQUEST_LOCK:
        now = time.monotonic()
        elapsed_ms = (now - _LAST_REQUEST_TIME) * 1000
        if elapsed_ms < cooldown_ms:
            time.sleep((cooldown_ms - elapsed_ms) / 1000)
        _LAST_REQUEST_TIME = time.monotonic()


def _request_audio_chunk(text, config, chunk_index, total_chunks):
    last_error = None
    for attempt in range(config["max_retries"]):
        try:
            _wait_for_request_slot(config["request_cooldown_ms"])
            response = requests.post(
                config["endpoint_url"],
                headers={"Content-Type": "application/json"},
                json={
                    "text": text,
                    "voice": config["voice"],
                    "model": config["model"],
                    "speed": config["speed"],
                },
                timeout=config["timeout_sec"],
            )
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            if not response.content:
                raise RuntimeError("Empty audio response")

            audio = AudioSegment.from_file(io.BytesIO(response.content))
            if len(audio) <= 0:
                raise RuntimeError("Decoded audio duration is 0")
            return audio
        except Exception as exc:
            last_error = exc
            rprint(
                f"[yellow]Proxy TTS chunk {chunk_index}/{total_chunks} failed "
                f"({attempt + 1}/{config['max_retries']}): {exc}[/yellow]"
            )
            if attempt < config["max_retries"] - 1:
                time.sleep(2**attempt)

    raise RuntimeError(f"Proxy TTS chunk {chunk_index}/{total_chunks} failed: {last_error}")


def proxy_tts(text, save_path):
    config = _load_proxy_config()
    if not config["endpoint_url"]:
        raise ValueError("proxy_tts.endpoint_url is required")
    chunks = split_text_into_chunks(text, config["chunk_size"])
    if not chunks:
        raise ValueError("Proxy TTS text is empty after chunking")

    rprint(f"[cyan]Proxy TTS generating {len(chunks)} chunk(s)[/cyan]")
    audio_segments = [
        _request_audio_chunk(chunk, config, index + 1, len(chunks))
        for index, chunk in enumerate(chunks)
    ]

    combined = AudioSegment.empty()
    for audio in audio_segments:
        combined += audio

    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(
        speech_file_path,
        format="wav",
        parameters=["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"],
    )

    if not os.path.exists(speech_file_path) or os.path.getsize(speech_file_path) == 0:
        raise RuntimeError(f"Proxy TTS output file is empty: {speech_file_path}")
    rprint(f"[green]Proxy TTS audio saved to {speech_file_path}[/green]")
