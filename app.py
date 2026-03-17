"""
GPU Efficiency Advisor using NVIDIA Nemotron
Hackathon MVP - single-file Streamlit app.
"""
from __future__ import annotations

import os
import streamlit as st
import requests

# Load .env from the same folder as this script (project root)
try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_script_dir, ".env"))
    if not os.environ.get("NVIDIA_API_KEY"):
        load_dotenv(os.path.join(_script_dir, ".env.example"))
except ImportError:
    pass  # python-dotenv not installed; use system env only

# ---------------------------------------------------------------------------
# Configuration - change model or endpoint here if needed
# ---------------------------------------------------------------------------
# NVIDIA NIM chat completions endpoint (OpenAI-compatible)
NEMOTRON_API_BASE = "https://integrate.api.nvidia.com/v1"
# Model: Nemotron 3 Super 120B; alternatives: nvidia/nemotron-3-nano-30b-a3b
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert GPU performance optimization advisor focused on NVIDIA workloads.
Your task is to analyze a user's AI training or inference configuration and provide practical recommendations to improve:
1. GPU utilization
2. Memory efficiency
3. Training or inference speed
4. Overall compute cost

When responding, use this format:

Bottleneck Summary:
- Briefly identify the main inefficiencies

Optimization Suggestions:
- Give 4 to 6 actionable suggestions
- Be practical and specific
- Mention techniques such as mixed precision, batch size tuning, gradient accumulation, distributed training, quantization, checkpointing, better GPU sizing, data pipeline improvements, or inference batching when relevant

Estimated Impact:
- Briefly describe likely improvements in utilization, speed, or cost

Important:
- Do not invent hard numbers unless clearly framed as estimates
- Keep the tone concise, technical, and demo-friendly
- If the input is incomplete, still provide best-effort recommendations based on the available details"""


def call_nemotron_chat(messages: list[dict], api_key: str) -> tuple[str | None, str | None]:
    """
    Send a chat-style request to NVIDIA Nemotron and return (content, error_message).
    Defensive parsing for varying response shapes.
    """
    url = f"{NEMOTRON_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NEMOTRON_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
        "stream": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        def extract_text(obj: dict) -> str | None:
            """Pull out a single string from message/content/parts."""
            if not obj:
                return None
            # Direct string fields
            for key in ("content", "text", "output"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                # Some APIs use content as list of {"type":"text","text":"..."}
                if isinstance(val, list):
                    parts = []
                    for item in val:
                        if isinstance(item, dict) and item.get("type") == "text":
                            t = item.get("text")
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
                        elif isinstance(item, str) and item.strip():
                            parts.append(item.strip())
                    if parts:
                        return "\n".join(parts)
            return None

        choices = data.get("choices")
        if choices and len(choices) > 0:
            choice = choices[0]
            msg = choice.get("message") or choice.get("delta") or choice
            out = extract_text(msg)
            if out:
                return (out, None)
        out = extract_text(data)
        if out:
            return (out, None)
        # Last resort: find any substantial string in the response (e.g. nested "text" or "content")
        def find_first_string(obj, min_len: int = 20) -> str | None:
            if isinstance(obj, str) and len(obj.strip()) >= min_len:
                return obj.strip()
            if isinstance(obj, dict):
                for v in obj.values():
                    found = find_first_string(v, min_len)
                    if found:
                        return found
            if isinstance(obj, list):
                for item in obj:
                    found = find_first_string(item, min_len)
                    if found:
                        return found
            return None
        fallback = find_first_string(data)
        if fallback:
            return (fallback, None)
        return (None, "API returned no content in the response.")
    except requests.exceptions.Timeout:
        return (None, "Request timed out. The API took too long to respond.")
    except requests.exceptions.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                err_body = e.response.json()
                detail = err_body.get("detail") or err_body.get("message") or err_body.get("error") or str(err_body)
                err_msg = f"{e}: {detail}"
            except Exception:
                err_msg = f"{e}. Response: {e.response.text[:300]}"
        return (None, err_msg)
    except (KeyError, IndexError, TypeError) as e:
        return (None, f"Unexpected API response format: {e}")


def analyze_gpu_config(user_input: str) -> tuple[str | None, str | None]:
    """
    Analyze the user's GPU config/log with NVIDIA Nemotron.
    Returns (result_text, error_message). One of them is None.
    """
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        return (None, "NVIDIA API key is not set.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input or "(No configuration provided.)"},
    ]
    content, err = call_nemotron_chat(messages, api_key)
    if content:
        return (content, None)
    return (None, err or "Analysis could not be completed. Please check your API key and try again.")


# ---------------------------------------------------------------------------
# Sample and example inputs for the UI
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE = """Model: Llama 13B
GPU: A100
Batch size: 4
GPU utilization: 42%
Memory usage: 68GB/80GB
Training time: 20 hours
Data loading occasionally stalls
Mixed precision: disabled
Distributed training: no"""

EXAMPLE_1 = """Model: Llama 13B
GPU: A100
Batch size: 4
GPU utilization: 42%
Memory usage: 68GB/80GB
Training time: 20 hours"""

EXAMPLE_2 = """Model: ResNet50
GPU: H100
Batch size: 8
GPU utilization: 35%
Memory usage: 22GB/80GB
Training time: 12 hours"""

EXAMPLE_3 = """Inference Model: 7B chatbot
GPU: A10
Average latency: 2.8s
Requests per second: 3
GPU utilization: 28%
Batching: disabled"""


def _rerun():
    """Rerun the app (works across Streamlit 1.28–1.50+)."""
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()


def main():
    st.set_page_config(page_title="GPU Efficiency Advisor", layout="centered")
    try:
        _main_content()
    except Exception as e:
        st.error("Something went wrong")
        st.exception(e)


def _main_content():
    # Ensure session_state keys exist (avoids white screen on first load)
    if "user_config" not in st.session_state:
        st.session_state["user_config"] = DEFAULT_SAMPLE

    # Apply "Use Example" before any widget that uses user_config (Streamlit rule)
    load_example = st.session_state.pop("load_example", None)
    if load_example == 1:
        st.session_state["user_config"] = EXAMPLE_1
    elif load_example == 2:
        st.session_state["user_config"] = EXAMPLE_2
    elif load_example == 3:
        st.session_state["user_config"] = EXAMPLE_3

    st.title("GPU Efficiency Advisor using NVIDIA Nemotron")
    st.caption(
        "Analyze your GPU training or inference configuration and get actionable "
        "optimization suggestions powered by NVIDIA Nemotron."
    )
    st.divider()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        st.warning(
            "**NVIDIA API key not set.** Add `NVIDIA_API_KEY` to your environment or `.env` file to enable analysis."
        )

    user_input = st.text_area(
        "Paste your GPU training config or logs",
        value=st.session_state["user_config"],
        height=220,
        placeholder="Paste configuration or log text here...",
        key="user_config",
    )

    # Show last analysis if we have one (persists across reruns)
    last = (st.session_state.get("last_analysis") or "").strip()
    if last:
        st.success("Analysis complete.")
        with st.container():
            st.markdown("**Analysis result:**")
            st.markdown(last.replace("\n", "  \n"))  # line breaks in markdown
        st.divider()

    if st.button("Analyze", type="primary", use_container_width=True):
        if not api_key:
            st.error("Cannot analyze: NVIDIA API key is missing.")
        else:
            with st.spinner("Calling NVIDIA Nemotron API… this may take 30–60 seconds."):
                result, err = analyze_gpu_config(user_input)
            if err:
                st.error(f"**Analysis failed:** {err}")
            elif result and result.strip():
                st.session_state["last_analysis"] = result.strip()
                st.success("Analysis complete.")
                st.markdown("**Analysis result:**")
                st.markdown(result.replace("\n", "  \n"))
            else:
                st.warning(
                    "The API returned an empty response. Your key may be invalid, or the model may not support this endpoint. "
                    "Check [NVIDIA API Catalog](https://build.nvidia.com) for the correct model and key."
                )

    st.divider()
    st.subheader("Example inputs")
    with st.expander("Example 1: Llama 13B on A100"):
        st.text(EXAMPLE_1)
        if st.button("Use Example 1", key="ex1"):
            st.session_state["load_example"] = 1
            _rerun()
    with st.expander("Example 2: ResNet50 on H100"):
        st.text(EXAMPLE_2)
        if st.button("Use Example 2", key="ex2"):
            st.session_state["load_example"] = 2
            _rerun()
    with st.expander("Example 3: 7B inference on A10"):
        st.text(EXAMPLE_3)
        if st.button("Use Example 3", key="ex3"):
            st.session_state["load_example"] = 3
            _rerun()

    st.divider()
    st.caption(
        "Built for NVIDIA Hackathon at SJSU | Powered by NVIDIA Nemotron"
    )


if __name__ == "__main__":
    main()
