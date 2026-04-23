import os
import json
import time
from threading import Lock
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.openai_compatible import (
    apply_degradation,
    build_request_url,
    build_request_urls,
    classify_retry_action,
    debug_log,
    load_cfg_safe,
    normalize_base_url,
    parse_json_relaxed,
    parse_response_text,
    post_openai_compatible,
    redact_payload,
    response_to_text,
    sanitize_payload,
)

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    model = load_key("api.model")
    api_key = load_key("api.key")
    api_protocol = load_cfg_safe(load_key, "api.api_protocol", "chat_completions")
    if api_protocol not in {"chat_completions", "responses"}:
        api_protocol = "chat_completions"

    compat_cfg = {
        "provider_type": load_cfg_safe(load_key, "api.provider_type", "openai_compatible"),
        "api_protocol": api_protocol,
        "supports_response_format": load_cfg_safe(load_key, "api.supports_response_format", load_key("api.llm_support_json")),
        "supports_tools": load_cfg_safe(load_key, "api.supports_tools", False),
        "sanitize_null_fields": load_cfg_safe(load_key, "api.sanitize_null_fields", True),
        "drop_optional_fields": load_cfg_safe(load_key, "api.drop_optional_fields", []),
        "auto_switch_protocol_on_block": load_cfg_safe(load_key, "api.auto_switch_protocol_on_block", True),
        "chat_completions_path": load_cfg_safe(load_key, "api.chat_completions_path", ""),
        "responses_path": load_cfg_safe(load_key, "api.responses_path", ""),
    }

    base_url = normalize_base_url(load_key("api.base_url"), api_protocol)
    timeout = int(load_cfg_safe(load_key, "api.request_timeout", 300))
    use_json_mode = resp_type == "json" and bool(load_key("api.llm_support_json"))
    response_format = {"type": "json_object"} if use_json_mode else None

    if api_protocol == "responses":
        params = {
            "model": model,
            "input": prompt,
            "timeout": timeout,
            "response_format": response_format,
        }
    else:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": response_format,
            "timeout": timeout,
        }
    params = sanitize_payload(params, compat_cfg)

    request_url = build_request_url(base_url, api_protocol)
    request_url_candidates = build_request_urls(base_url, api_protocol, compat_cfg)
    debug_log(
        "request_prepared",
        {
            "log_title": log_title,
            "provider": compat_cfg["provider_type"],
            "protocol": api_protocol,
            "base_url": base_url,
            "request_url": request_url,
            "request_url_candidates": request_url_candidates,
            "payload": redact_payload(params),
        },
    )

    max_retry = 5
    degraded_steps = 0
    protocol_switched = False
    last_exc = None
    for attempt in range(max_retry):
        try:
            resp_raw, status_code, raw_text, final_url = post_openai_compatible(
                base_url=base_url,
                api_key=api_key,
                api_protocol=api_protocol,
                payload=params,
                timeout=timeout,
                cfg=compat_cfg,
            )

            resp_content, parser_path = parse_response_text(resp_raw)
            parse_attempts = ["text"]
            if resp_type == "json":
                resp, parse_attempts = parse_json_relaxed(resp_content)
            else:
                resp = resp_content

            if valid_def:
                valid_resp = valid_def(resp)
                if valid_resp['status'] != 'success':
                    _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
                    raise ValueError(f"❎ API response error: {valid_resp['message']}")

            debug_log(
                "request_success",
                {
                    "log_title": log_title,
                    "protocol": api_protocol,
                    "base_url": base_url,
                    "request_url": final_url,
                    "status_code": status_code,
                    "parser_path": parser_path,
                    "json_parse_attempts": parse_attempts,
                    "raw_response": raw_text or response_to_text(resp_raw, max_len=2000),
                },
            )
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
            return resp
        except Exception as exc:
            last_exc = exc
            error_text = str(exc)
            debug_log(
                "request_error",
                {
                    "log_title": log_title,
                    "protocol": api_protocol,
                    "base_url": base_url,
                    "request_url": build_request_url(base_url, api_protocol),
                    "request_url_candidates": build_request_urls(base_url, api_protocol, compat_cfg),
                    "retry": attempt + 1,
                    "error": error_text,
                    "payload": redact_payload(params),
                },
            )

            if attempt == max_retry - 1:
                break

            action = classify_retry_action(exc, degraded_steps)
            if action:
                params = apply_degradation(params, action, compat_cfg)
                degraded_steps += 1
                debug_log(
                    "request_degraded",
                    {
                        "log_title": log_title,
                        "action": action,
                        "retry": attempt + 1,
                        "payload": redact_payload(params),
                    },
                )
                rprint(f"[yellow]Relay compatibility fallback: {action}[/yellow]")

            lowered_err = error_text.lower()
            blocked_like = (
                "blocked" in lowered_err
                or "http 403" in lowered_err
                or "http 502" in lowered_err
                or "http 503" in lowered_err
                or "http 504" in lowered_err
            )
            if (
                not action
                and compat_cfg.get("auto_switch_protocol_on_block", True)
                and blocked_like
                and not protocol_switched
            ):
                api_protocol = "chat_completions" if api_protocol == "responses" else "responses"
                compat_cfg["api_protocol"] = api_protocol
                protocol_switched = True
                if api_protocol == "responses":
                    params = {
                        "model": model,
                        "input": prompt,
                        "response_format": response_format,
                        "timeout": timeout,
                    }
                else:
                    params = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": response_format,
                        "timeout": timeout,
                    }
                params = sanitize_payload(params, compat_cfg)
                debug_log(
                    "request_protocol_switched",
                    {
                        "log_title": log_title,
                        "new_protocol": api_protocol,
                        "request_url": build_request_url(base_url, api_protocol),
                        "request_url_candidates": build_request_urls(base_url, api_protocol, compat_cfg),
                        "retry": attempt + 1,
                        "payload": redact_payload(params),
                    },
                )
                rprint(f"[yellow]Relay compatibility fallback: switch protocol to {api_protocol}[/yellow]")

            rprint(f"[red]GPT request failed: {error_text}, retry: {attempt+1}/{max_retry}[/red]")
            time.sleep(2 ** attempt)

    raise last_exc


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
