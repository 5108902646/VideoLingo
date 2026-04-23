import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse, urlunparse

import json_repair
import requests


DEBUG_LOG_FILE = "output/gpt_log/relay_compat_debug.jsonl"


def load_cfg_safe(load_key_fn, key, default=None):
    try:
        return load_key_fn(key)
    except KeyError:
        return default


def _strip_trailing_v1_path(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunparse(parsed._replace(path=path or ""))


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

    # Keep the URL as entered (after normalization) and probe both with/without /v1 later.
    return url


def _with_v1_path(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunparse(parsed._replace(path=path))


def build_base_url_candidates(base_url: str):
    raw = (base_url or "").rstrip("/")
    with_v1 = _with_v1_path(raw).rstrip("/")
    without_v1 = _strip_trailing_v1_path(raw).rstrip("/")

    candidates = []
    for item in [raw, with_v1, without_v1]:
        if item and item not in candidates:
            candidates.append(item)
    return candidates


def build_request_urls(base_url: str, api_protocol: str, cfg: dict | None = None):
    cfg = cfg or {}
    if api_protocol == "responses":
        custom_path = (cfg.get("responses_path") or "").strip()
        endpoint_paths = [
            custom_path,
            "/responses",
            "/v1/responses",
            "/api/v1/responses",
            "/openai/v1/responses",
        ]
    else:
        custom_path = (cfg.get("chat_completions_path") or "").strip()
        endpoint_paths = [
            custom_path,
            "/chat/completions",
            "/v1/chat/completions",
            "/api/v1/chat/completions",
            "/openai/v1/chat/completions",
        ]

    normalized_paths = []
    for path in endpoint_paths:
        if not path:
            continue
        if not path.startswith("/"):
            path = "/" + path
        if path not in normalized_paths:
            normalized_paths.append(path)

    urls = []
    for base in build_base_url_candidates(base_url):
        for path in normalized_paths:
            url = f"{base.rstrip('/')}{path}"
            if url not in urls:
                urls.append(url)
    return urls


def build_request_url(base_url: str, api_protocol: str):
    # Keep backward-compatible single-URL helper for existing logs.
    return build_request_urls(base_url, api_protocol)[0]


def build_models_urls(base_url: str, cfg: dict | None = None):
    cfg = cfg or {}
    custom_path = (cfg.get("models_path") or "").strip()
    default_path = custom_path if custom_path else "/models"
    if not default_path.startswith("/"):
        default_path = "/" + default_path

    urls = []
    for base in build_base_url_candidates(base_url):
        model_url = f"{base.rstrip('/')}{default_path}"
        if model_url not in urls:
            urls.append(model_url)
    return urls


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


def post_openai_compatible(base_url: str, api_key: str, api_protocol: str, payload: dict, timeout: int = 300, cfg: dict | None = None):
    candidate_urls = build_request_urls(base_url, api_protocol, cfg)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None

    def _is_probably_html(text):
        if not isinstance(text, str):
            return False
        low = text.lower()
        return "<html" in low or "<!doctype html" in low

    def _is_llm_payload(data):
        if isinstance(data, dict):
            # Chat Completions-like
            if isinstance(data.get("choices"), list) and data.get("choices"):
                return True
            # Responses-like
            if "output" in data or "output_text" in data:
                return True
            # Common OpenAI shape hint
            if isinstance(data.get("object"), str) and data.get("object") in {
                "chat.completion",
                "response",
            }:
                return True
            # Explicit error payload should not be treated as success.
            if "error" in data:
                return False
            # Known website config payload from Sub2API dashboard page.
            if "site_name" in data and "site_subtitle" in data:
                return False
        return False

    for url in candidate_urls:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        raw_text = truncate_text(resp.text or "", 2000)

        parsed = None
        if resp.text:
            try:
                parsed = resp.json()
            except Exception:
                parsed = resp.text

        if resp.status_code < 400:
            # Some relays return website HTML (200 OK) for wrong endpoint paths.
            if _is_probably_html(raw_text):
                last_error = {
                    "status_code": resp.status_code,
                    "error": "Received HTML page instead of LLM API response",
                    "url": url,
                }
                continue
            if _is_llm_payload(parsed):
                return parsed, resp.status_code, raw_text, url
            last_error = {
                "status_code": resp.status_code,
                "error": "Endpoint responded but payload is not an LLM schema",
                "url": url,
            }
            continue

        err_msg = raw_text
        if isinstance(parsed, dict):
            error_obj = parsed.get("error")
            if isinstance(error_obj, dict):
                err_msg = error_obj.get("message") or error_obj.get("code") or raw_text
        elif isinstance(parsed, str):
            title_match = re.search(r"<title>(.*?)</title>", parsed, re.IGNORECASE | re.DOTALL)
            if title_match:
                err_msg = title_match.group(1).strip()
            elif "bad gateway" in parsed.lower():
                err_msg = "Bad gateway"

        if resp.status_code in {502, 503, 504}:
            err_msg = f"Relay upstream gateway error ({err_msg})"

        last_error = {
            "status_code": resp.status_code,
            "error": err_msg,
            "url": url,
        }

        # Keep probing other candidates for routing/path issues.
        if resp.status_code in {404, 405, 502, 503, 504}:
            continue
        raise ValueError(f"HTTP {resp.status_code}: {err_msg} @ {url}")

    if last_error:
        raise ValueError(
            f"HTTP {last_error['status_code']}: {last_error['error']} @ {last_error['url']}"
        )
    raise ValueError("Relay request failed without response")
