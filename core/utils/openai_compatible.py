import json
import os
from datetime import datetime
from urllib.parse import urlparse, urlunparse

import json_repair


DEBUG_LOG_FILE = "output/gpt_log/relay_compat_debug.jsonl"


def load_cfg_safe(load_key_fn, key, default=None):
    try:
        return load_key_fn(key)
    except KeyError:
        return default


def _ensure_v1_path(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        new_path = path
    elif "/v1/" in f"{path}/":
        new_path = path
    elif not path:
        new_path = "/v1"
    else:
        new_path = f"{path}/v1"
    return urlunparse(parsed._replace(path=new_path))


def normalize_base_url(base_url: str, api_protocol: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("API base_url is empty")

    # Keep existing special handling for Volcengine Ark.
    if "ark" in url:
        return "https://ark.cn-beijing.volces.com/api/v3"

    # Accept users pasting full endpoint URLs and normalize back to base /v1.
    for suffix in ["/chat/completions", "/responses"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break

    # OpenAI-compatible relays usually expect /v1 for both chat and responses APIs.
    if api_protocol in {"chat_completions", "responses"}:
        return _ensure_v1_path(url).rstrip("/")
    return url


def sanitize_payload(payload: dict, cfg: dict) -> dict:
    cleaned = dict(payload)
    if cfg.get("sanitize_null_fields", True):
        cleaned = {k: v for k, v in cleaned.items() if v is not None}

    if not cfg.get("supports_response_format", False):
        cleaned.pop("response_format", None)
    if not cfg.get("supports_tools", False):
        cleaned.pop("tools", None)
        cleaned.pop("parallel_tool_calls", None)

    drop_list = cfg.get("drop_optional_fields") or []
    for field in drop_list:
        cleaned.pop(field, None)

    return cleaned


def _extract_from_content_parts(parts):
    texts = []
    for item in parts:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            if item.get("type") in {"text", "output_text"} and item.get("text"):
                texts.append(item.get("text"))
            elif isinstance(item.get("text"), str):
                texts.append(item.get("text"))
            elif isinstance(item.get("content"), str):
                texts.append(item.get("content"))
        else:
            text_val = getattr(item, "text", None)
            if isinstance(text_val, str):
                texts.append(text_val)
    return "\n".join([t for t in texts if t]).strip()


def _get_output_items(resp_obj):
    output = getattr(resp_obj, "output", None)
    if output is None and isinstance(resp_obj, dict):
        output = resp_obj.get("output")
    return output if isinstance(output, list) else []


def parse_response_text(resp_obj):
    paths = []

    choices = getattr(resp_obj, "choices", None)
    if choices is None and isinstance(resp_obj, dict):
        choices = resp_obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content, "choices[0].message.content"
            if isinstance(content, list):
                merged = _extract_from_content_parts(content)
                if merged:
                    return merged, "choices[0].message.content[]"
        text = getattr(first, "text", None)
        if text is None and isinstance(first, dict):
            text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text, "choices[0].text"
        paths.append("choices")

    output_text = getattr(resp_obj, "output_text", None)
    if output_text is None and isinstance(resp_obj, dict):
        output_text = resp_obj.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text, "output_text"

    output_items = _get_output_items(resp_obj)
    if output_items:
        for idx, item in enumerate(output_items):
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if isinstance(content, list):
                merged = _extract_from_content_parts(content)
                if merged:
                    return merged, f"output[{idx}].content[]"
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if isinstance(text, str) and text.strip():
                return text, f"output[{idx}].text"
        paths.append("output")

    if isinstance(resp_obj, str) and resp_obj.strip():
        return resp_obj, "raw_string"

    raise ValueError(f"Unable to extract content text from response. parsed_paths={paths}")


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return value


def _extract_first_json_block(text: str):
    candidates = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    if not candidates:
        return None
    return min(candidates, key=len) if len(candidates) > 1 else candidates[0]


def parse_json_relaxed(text: str):
    attempts = []
    errors = []

    def _try_parse(label: str, candidate: str):
        attempts.append(label)
        try:
            return json_repair.loads(candidate)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            return None

    parsed = _try_parse("raw", text)
    if parsed is not None:
        return parsed, attempts

    stripped = _strip_code_fence(text)
    if stripped != text:
        parsed = _try_parse("code_fence_stripped", stripped)
        if parsed is not None:
            return parsed, attempts

    block = _extract_first_json_block(stripped)
    if block:
        parsed = _try_parse("first_json_block", block)
        if parsed is not None:
            return parsed, attempts

    raise ValueError("JSON parse failed. attempts=%s errors=%s" % (attempts, errors))


def redact_payload(payload: dict):
    if not isinstance(payload, dict):
        return payload
    redacted = {}
    for key, value in payload.items():
        if key.lower() in {"api_key", "authorization", "token"}:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


def truncate_text(value: str, max_len: int):
    if not isinstance(value, str):
        value = str(value)
    return value[:max_len]


def debug_log(event: str, payload: dict):
    os.makedirs(os.path.dirname(DEBUG_LOG_FILE), exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        "payload": payload,
    }
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_retry_action(exc: Exception, degraded_steps: int):
    msg = str(exc).lower()
    if degraded_steps >= 3:
        return None
    if "unexpected keyword argument 'messages'" in msg:
        return "fix_protocol_payload"
    if any(flag in msg for flag in ["blocked", "unsupported", "invalid_request_error", "bad request", "400"]):
        if degraded_steps == 0:
            return "drop_response_format"
        if degraded_steps == 1:
            return "drop_optional_fields"
        return "minimal_payload"
    return None


def apply_degradation(payload: dict, action: str, cfg: dict):
    protocol = cfg.get("api_protocol", "chat_completions")

    if action == "fix_protocol_payload":
        if protocol == "responses":
            payload.pop("messages", None)
        else:
            payload.pop("input", None)
    if action == "drop_response_format":
        payload.pop("response_format", None)
    elif action == "drop_optional_fields":
        for key in ["tools", "parallel_tool_calls", "seed", "temperature", "top_p", "presence_penalty", "frequency_penalty"]:
            payload.pop(key, None)
    elif action == "minimal_payload":
        if protocol == "responses":
            payload = {
                "model": payload.get("model"),
                "timeout": payload.get("timeout", 300),
                "input": payload.get("input"),
            }
        else:
            payload = {
                "model": payload.get("model"),
                "timeout": payload.get("timeout", 300),
                "messages": payload.get("messages", []),
            }
        payload = {k: v for k, v in payload.items() if v is not None}
    return sanitize_payload(payload, cfg)


def response_to_text(resp_obj, max_len=2000):
    try:
        if hasattr(resp_obj, "model_dump_json"):
            return truncate_text(resp_obj.model_dump_json(), max_len)
        if isinstance(resp_obj, dict):
            return truncate_text(json.dumps(resp_obj, ensure_ascii=False), max_len)
        return truncate_text(str(resp_obj), max_len)
    except Exception:
        return truncate_text(str(resp_obj), max_len)
