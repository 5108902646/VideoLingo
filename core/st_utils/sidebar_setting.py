import streamlit as st
import requests
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *
from core.utils.openai_compatible import load_cfg_safe, build_models_urls


def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


def _load_key_safe(key, default=None):
    return load_cfg_safe(load_key, key, default)


def _fetch_model_list(base_url, api_key):
    """Fetch available models from OpenAI-compatible /v1/models endpoint."""
    if not api_key or not base_url:
        return []
    models_path = _load_key_safe("api.models_path", "")
    cfg = {"models_path": models_path}
    candidate_urls = build_models_urls(base_url, cfg)
    headers = {"Authorization": f"Bearer {api_key}"}

    for url in candidate_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code >= 400:
                continue
            body = resp.json()
            data = body.get("data", []) if isinstance(body, dict) else []
            models = sorted([m["id"] for m in data if isinstance(m, dict) and "id" in m])
            if models:
                return models
        except Exception:
            continue
    return []


def _probe_models_auth(base_url, api_key):
    """Probe /models style endpoints to classify auth vs gateway/network issues."""
    models_path = _load_key_safe("api.models_path", "")
    cfg = {"models_path": models_path}
    candidate_urls = build_models_urls(base_url, cfg)
    headers = {"Authorization": f"Bearer {api_key}"}
    saw_gateway_error = False
    last_error = ""

    for url in candidate_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code in (200, 201):
                return {"status": "ok", "message": "Models endpoint reachable"}
            if resp.status_code in (401, 403):
                return {"status": "auth_error", "message": f"HTTP {resp.status_code} from models endpoint"}
            if resp.status_code in (502, 503, 504):
                saw_gateway_error = True
                last_error = f"HTTP {resp.status_code} from models endpoint"
                continue
            if resp.status_code in (404, 405):
                continue
            last_error = f"HTTP {resp.status_code} from models endpoint"
        except Exception as e:
            last_error = str(e)

    if saw_gateway_error:
        return {"status": "relay_error", "message": last_error or "Relay upstream gateway error"}
    if last_error:
        return {"status": "unknown", "message": last_error}
    return {"status": "unknown", "message": "No reachable models endpoint"}


def _classify_api_check_error(exc):
    msg = str(exc)
    low = msg.lower()
    if any(k in low for k in ["http 401", "http 403", "unauthorized", "invalid api key", "incorrect api key"]):
        return {"ok": False, "code": "auth_error", "message": t("API Key is invalid")}
    if any(k in low for k in ["http 502", "http 503", "http 504", "bad gateway", "upstream gateway"]):
        return {"ok": False, "code": "relay_error", "message": "Relay is temporarily unavailable (5xx). API key may still be valid."}
    if any(k in low for k in ["timeout", "timed out", "connection error", "name or service not known"]):
        return {"ok": False, "code": "network_error", "message": "Network or DNS error while checking API."}
    if "http 400" in low and "model" in low:
        return {"ok": False, "code": "model_error", "message": "API reachable, but model is invalid or unsupported."}
    return {"ok": False, "code": "unknown", "message": f"API check failed: {msg}"}


def _search_models(search_term, **kwargs):
    """Search function for st_searchbox — returns models matching the search term."""
    models = st.session_state.get("_model_list", [])
    if not search_term:
        return models if models else []
    term = search_term.lower()
    matched = [m for m in models if term in m.lower()]
    # Always include the raw input as an option so users can type custom model names
    if search_term not in matched:
        matched.insert(0, search_term)
    return matched


def page_setting():
    # Widen the sidebar slightly to accommodate the model searchbox
    st.markdown(
        """<style>[data-testid="stSidebar"] {min-width: 420px; max-width: 420px;}</style>""",
        unsafe_allow_html=True,
    )

    display_language = st.selectbox(
        "Display Language 🌐",
        options=list(DISPLAY_LANGUAGES.keys()),
        index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")),
    )
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key", placeholder=t("Enter your API key"))
        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("OpenAI-compatible base URL, supports host / host/ / host/v1"),
        )

        # Try to use searchbox for model selection, fall back to text_input
        try:
            from streamlit_searchbox import st_searchbox
            from streamlit_searchbox import _list_to_options_js, _list_to_options_py

            if st.button(
                t("Fetch Model List"), key="fetch_models", use_container_width=True
            ):
                with st.spinner(t("Fetching models...")):
                    models = _fetch_model_list(
                        load_key("api.base_url"), load_key("api.key")
                    )
                    st.session_state["_model_list"] = models
                    if models:
                        # Update searchbox internal state directly so dropdown shows options
                        sb_key = "model_searchbox"
                        if sb_key in st.session_state:
                            st.session_state[sb_key]["options_js"] = (
                                _list_to_options_js(models)
                            )
                            st.session_state[sb_key]["options_py"] = (
                                _list_to_options_py(models)
                            )
                        st.toast(
                            t("Fetched {n} models").replace("{n}", str(len(models))),
                            icon="✅",
                        )
                    else:
                        st.toast(
                            t(
                                "Failed to fetch models, please check API Key and Base URL"
                            ),
                            icon="❌",
                        )

            current_model = load_key("api.model")
            model_list = st.session_state.get("_model_list", None)

            sb_key = "model_searchbox"
            selected = st_searchbox(
                _search_models,
                placeholder=t("Search or enter model name"),
                default=current_model if current_model else None,
                default_searchterm=current_model if current_model else "",
                default_use_searchterm=True,
                default_options=model_list if model_list else None,
                key=sb_key,
                clear_on_submit=False,
            )
            if selected and selected != load_key("api.model"):
                update_key("api.model", selected)

            if st.button("📡 " + t("Check API"), key="api", use_container_width=True):
                with st.spinner(t("Check API") + "..."):
                    check_result = check_api()
                st.toast(
                    check_result.get("message", t("API Key is invalid")),
                    icon="✅" if check_result.get("ok") else "❌",
                )
        except ImportError:
            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(
                    t("MODEL"),
                    "api.model",
                    help=t("click to check API validity") + " 👉",
                    placeholder=t("Search or enter model name"),
                )
            with c2:
                if st.button("📡", key="api"):
                    check_result = check_api()
                    st.toast(
                        check_result.get("message", t("API Key is invalid")),
                        icon="✅" if check_result.get("ok") else "❌",
                    )
        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()

        with st.expander(t("Relay Compatibility (Advanced)"), expanded=False):
            api_protocol = st.selectbox(
                t("API Protocol"),
                options=["chat_completions", "responses"],
                index=["chat_completions", "responses"].index(_load_key_safe("api.api_protocol", "chat_completions")),
                help=t("Choose relay protocol path strategy"),
            )
            if api_protocol != _load_key_safe("api.api_protocol", "chat_completions"):
                update_key("api.api_protocol", api_protocol)
                st.rerun()

            supports_response_format = st.toggle(
                t("Supports response_format"),
                value=_load_key_safe("api.supports_response_format", False),
                help=t("Turn off for relays that block JSON mode parameters"),
            )
            if supports_response_format != _load_key_safe("api.supports_response_format", False):
                update_key("api.supports_response_format", supports_response_format)
                st.rerun()

            supports_tools = st.toggle(
                t("Supports tools/advanced params"),
                value=_load_key_safe("api.supports_tools", False),
                help=t("Turn off for strict OpenAI-compatible relays"),
            )
            if supports_tools != _load_key_safe("api.supports_tools", False):
                update_key("api.supports_tools", supports_tools)
                st.rerun()

            sanitize_null_fields = st.toggle(
                t("Sanitize null/None fields"),
                value=_load_key_safe("api.sanitize_null_fields", True),
            )
            if sanitize_null_fields != _load_key_safe("api.sanitize_null_fields", True):
                update_key("api.sanitize_null_fields", sanitize_null_fields)
                st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja",
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language")),
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtime = st.selectbox(
            t("WhisperX Runtime"),
            options=["local", "cloud", "elevenlabs"],
            index=["local", "cloud", "elevenlabs"].index(load_key("whisper.runtime")),
            help=t(
                "Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key"
            ),
        )
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t(
                    "Input any language in natural language, as long as llm can understand"
                ),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            help=t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        burn_subtitles = st.toggle(
            t("Burn-in Subtitles"),
            value=load_key("burn_subtitles"),
            help=t(
                "Whether to burn subtitles into the video, will increase processing time"
            ),
        )
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_methods = [
            "azure_tts",
            "openai_tts",
            "fish_tts",
            "sf_fish_tts",
            "edge_tts",
            "gpt_sovits",
            "custom_tts",
            "sf_cosyvoice2",
            "f5tts",
        ]
        select_tts = st.selectbox(
            t("TTS Method"),
            options=tts_methods,
            index=tts_methods.index(load_key("tts_method")),
        )
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            fish_tts_character = st.selectbox(
                t("Fish TTS Character"),
                options=list(load_key("fish_tts.character_id_dict").keys()),
                index=list(load_key("fish_tts.character_id_dict").keys()).index(
                    load_key("fish_tts.character")
                ),
            )
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(
                    load_key("gpt_sovits.refer_mode")
                ),
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")

        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")


def check_api():
    # Step 1: auth probe via models endpoint to avoid false invalid-key on relay 5xx.
    probe = _probe_models_auth(load_key("api.base_url"), load_key("api.key"))
    if probe["status"] == "ok":
        return {"ok": True, "message": t("API Key is valid")}
    if probe["status"] == "auth_error":
        return {"ok": False, "message": t("API Key is invalid")}

    # Step 2: functional check via completion call.
    try:
        resp = ask_gpt(
            "Reply with exactly: success",
            resp_type=None,
            log_title="None",
        )
        ok = isinstance(resp, str) and "success" in resp.lower()
        if ok:
            return {"ok": True, "message": t("API Key is valid")}
        return {"ok": False, "message": "API reachable, but test response is unexpected."}
    except Exception as e:
        return _classify_api_check_error(e)


if __name__ == "__main__":
    check_api()
