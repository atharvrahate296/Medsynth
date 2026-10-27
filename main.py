import html
import io
import os
import random
import re
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Offline mode: always use local fine-tuned weights, never connect to HF Hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import streamlit as st
from PIL import Image

from utils import chat_history, prompt_enhancer
from utils.prompt_enhancer import IMAGE_TYPES

# ---------------------------------------------------------------------------
# Project & Model Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("MEDSYNTH_MODEL_DIR", PROJECT_DIR / "medsynth-model" / "pipeline")
)
_generation_lock = threading.Lock()

IMAGE_TYPES_DICT = dict(IMAGE_TYPES)

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, anime, sketch, "
    "3d render, CGI, digital art, artificial skin, plastic skin, "
    "waxy skin, excessively smooth skin, fake texture, "
    "unrealistic pigmentation, neon colors, oversaturated colors, "
    "extreme color grading, dramatic lighting, cinematic lighting, "
    "glowing lesion, excessive contrast, "
    "blur, motion blur, out of focus, low resolution, "
    "compression artifacts, noise, distorted anatomy, "
    "deformed body, duplicated body parts, malformed lesion, "
    "unnatural symmetry, impossible anatomy, "
    "watermark, text, logo, labels, arrows, "
    "UI, border, frame, "
    "food, landscape, scenery"
)

DEFAULT_SETTINGS = dict(chat_history.SETTINGS_DEFAULTS)
DEFAULT_SETTINGS["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
DEFAULT_SETTINGS["clinical"] = True

ENHANCEMENT_FOOTER_NOTE = (
    "The fine-tune was trained on short captions in a fixed grammar: "
    "`<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick type>`. "
    "Clinical enhancement parses your description, normalizes it into this grammar, "
    "infers missing clinical elements, and appends the appropriate imaging terms "
    "within CLIP's 75-token budget."
)


def get_effective_model_dir() -> Path:
    if (MODEL_DIR / "model_index.json").is_file():
        return MODEL_DIR
    fallback = PROJECT_DIR / "medsynth-model" / "pipeline"
    if (fallback / "model_index.json").is_file():
        return fallback
    fallback_model = PROJECT_DIR / "model" / "pipeline"
    if (fallback_model / "model_index.json").is_file():
        return fallback_model
    return MODEL_DIR


def model_is_available() -> bool:
    return (get_effective_model_dir() / "model_index.json").is_file()


_tokenizer_dir = get_effective_model_dir() / "tokenizer"
if _tokenizer_dir.is_dir():
    prompt_enhancer.set_tokenizer_dir(_tokenizer_dir)


# ---------------------------------------------------------------------------
# Pipeline Loader (Cached across Streamlit reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the fine-tuned Stable Diffusion pipeline into GPU/CPU memory once."""
    target_dir = get_effective_model_dir()
    if not (target_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Trained model not found at {target_dir}. Run train.py first "
            "or set MEDSYNTH_MODEL_DIR to the directory containing model_index.json."
        )

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        str(target_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )

    if device == "cuda":
        try:
            pipeline = pipeline.to(device)
            pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipeline.enable_attention_slicing()
            try:
                pipeline.enable_model_cpu_offload()
            except Exception:
                pipeline = pipeline.to("cpu")
                device = "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return pipeline, device


def get_device_info():
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
            return True, f"GPU Ready · {name} · FP16"
        return False, "CPU Mode · FP32 (slower)"
    except Exception as e:
        return False, f"Device info unavailable ({e})"


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------
def _relative_stamp(value):
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return ""
    delta = datetime.now() - moment
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return moment.strftime("%d %b")
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 86400 * 7:
        return f"{seconds // 86400}d ago"
    return moment.strftime("%d %b")


def _short_title(title, max_len=26):
    title = " ".join((title or "New conversation").split())
    if len(title) > max_len:
        return title[: max_len - 1].rstrip() + "…"
    return title


def _metadata_caption(plan, image_type, steps, guidance, used_seed):
    caption = plan.as_caption() if plan.enhanced else (plan.text or "custom prompt")
    facts = [
        f"{int(steps)} steps",
        f"CFG {float(guidance):g}",
        f"seed {used_seed}",
        f"{plan.tokens} tokens",
    ]
    if not plan.enhanced:
        facts.insert(0, "verbatim")
    elif plan.truncated:
        facts.insert(0, "trimmed")

    return (
        f'<div class="ms-cap">'
        f"<b>{html.escape(image_type)}</b> · {html.escape(caption)}<br>"
        f'<span class="ms-cap__meta">{html.escape(" · ".join(facts))}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Streamlit App Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medsynth | Dermoscopy Image Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# De-Cluttered Clinical Light Theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
    --primary: #0d9488;
    --primary-hover: #0f766e;
    --primary-light: #f0fdfa;
    --primary-border: #99f6e4;
    --text-main: #0f172a;
    --text-muted: #475569;
    --text-subtle: #64748b;
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --border-color: #e2e8f0;
    --border-focus: #0d9488;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.03);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
}

html, body, [class*="css"] {
    font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text-main);
    background-color: var(--bg-page);
}

/* Page container top padding reduction to eliminate empty top space */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1160px !important;
}

/* Clean, Spacious Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    padding: 1.2rem 1rem !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

/* Brand header */
.brand-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.6rem;
}
.brand-icon {
    font-size: 22px;
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    border-radius: 10px;
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    color: #0d9488;
}
.brand-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.15;
    margin: 0;
}
.brand-subtitle {
    font-size: 10.5px;
    color: #0d9488;
    font-weight: 600;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Hardware status badge */
.hw-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 10.5px;
    font-weight: 600;
    margin-bottom: 0.8rem;
    width: 100%;
    justify-content: center;
}
.hw-pill--gpu {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.hw-pill--cpu {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}
.hw-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}

/* Clean Section Headers */
.sidebar-section-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin: 1rem 0 0.4rem 0;
}

/* Clean, Non-Clustered Conversation Buttons */
.stButton button[kind="primary"] {
    background-color: var(--primary) !important;
    color: #ffffff !important;
    border: 1px solid var(--primary) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    box-shadow: 0 2px 4px rgba(13, 148, 136, 0.15) !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton button[kind="primary"]:hover {
    background-color: var(--primary-hover) !important;
    border-color: var(--primary-hover) !important;
    transform: translateY(-1px) !important;
}

.stButton button[kind="secondary"],
.stButton button:not([kind="primary"]) {
    background-color: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    transition: all 0.12s ease !important;
}
.stButton button[kind="secondary"]:hover,
.stButton button:not([kind="primary"]):hover {
    background-color: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Danger Button */
.danger-btn button {
    background-color: #fff1f2 !important;
    color: #be123c !important;
    border: 1px solid #fecdd3 !important;
    font-weight: 600 !important;
}

/* Studio Header (Main Content) */
.studio-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: var(--radius-md);
    padding: 14px 20px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-sm);
}
.studio-title-area h2 {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 2px 0;
}
.studio-title-area p {
    font-size: 12px;
    color: #64748b;
    margin: 0;
}

/* Clinical Badge */
.clinical-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    border-radius: 999px;
    color: #0f766e;
    font-size: 11px;
    font-weight: 600;
}
.clinical-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #0d9488;
}

/* Token Cost Meter */
.token-meter-container {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: var(--radius-sm);
    padding: 5px 12px;
    font-size: 11.5px;
    color: #334155;
}
.token-meter-bar {
    width: 48px;
    height: 6px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}
.token-meter-fill {
    height: 100%;
    background: #0d9488;
    border-radius: inherit;
    transition: width 0.2s ease;
}
.token-meter-fill.warn {
    background: #f59e0b;
}

/* User Message Bubble */
.user-msg-bubble {
    background: #0f172a;
    color: #ffffff !important;
    padding: 12px 18px;
    border-radius: 14px 14px 3px 14px;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    font-size: 13.5px;
    line-height: 1.45;
}
.user-msg-bubble strong {
    color: #5eead4;
}

/* Assistant Message Card */
.assistant-msg-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    margin-right: auto;
    max-width: 88%;
    margin-bottom: 18px;
    box-shadow: var(--shadow-sm);
}
.ms-cap {
    margin-top: 10px;
    padding: 8px 12px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #f8fafc;
    color: #1e293b;
    font-size: 11.5px;
    line-height: 1.4;
}
.ms-cap__meta {
    color: #64748b;
    font-size: 10.5px;
    display: inline-block;
    margin-top: 2px;
}

/* Clean Empty State Welcome Card */
.empty-state-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: var(--radius-md);
    padding: 24px 20px 20px;
    text-align: center;
    margin: 16px auto 14px;
    max-width: 620px;
    box-shadow: var(--shadow-sm);
}
.empty-state-title {
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.empty-state-desc {
    font-size: 12.5px;
    color: #64748b;
    margin-bottom: 14px;
    line-height: 1.4;
}
.prompt-chips-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: #475569;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

/* Clean Prompt Suggestion Button */
.sample-prompt-btn {
    text-align: left !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    margin-bottom: 6px !important;
    color: #0f172a !important;
    font-size: 12px !important;
    font-family: monospace !important;
}

/* Custom inputs styling */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.12) !important;
}

/* Radio buttons horizontal styling */
div[data-testid="stRadio"] > div {
    gap: 8px;
}
div[data-testid="stRadio"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* Expander styling clean */
div[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}

/* Studio Footer */
.studio-footer {
    margin-top: 36px;
    padding: 14px 18px;
    border-top: 1px solid #e2e8f0;
    background: #ffffff;
    border-radius: var(--radius-sm);
    color: #64748b;
    font-size: 11.5px;
    line-height: 1.5;
    box-shadow: var(--shadow-sm);
}
.studio-footer code {
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    color: #0f766e;
    font-size: 11px;
}
.footer-subtext {
    margin-top: 5px;
    font-size: 10.5px;
    color: #94a3b8;
    text-align: right;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    all_convs = chat_history.list_conversations()
    if not all_convs:
        initial_id = chat_history.create_conversation()
    else:
        initial_id = all_convs[0]["id"]
    st.session_state.conversation_id = initial_id

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None

if "prompt_draft" not in st.session_state:
    st.session_state.prompt_draft = ""

# Ensure active conversation exists
current_conv = chat_history.get_conversation(st.session_state.conversation_id)
if not current_conv:
    all_convs = chat_history.list_conversations()
    if all_convs:
        st.session_state.conversation_id = all_convs[0]["id"]
        current_conv = all_convs[0]
    else:
        new_id = chat_history.create_conversation()
        st.session_state.conversation_id = new_id
        current_conv = chat_history.get_conversation(new_id)

current_settings = chat_history.get_settings(st.session_state.conversation_id)
if not current_settings.get("negative_prompt"):
    current_settings["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT

# ---------------------------------------------------------------------------
# SIDEBAR: Streamlined & De-Cluttered Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. Brand & Title
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon">🔬</div>
            <div>
                <p class="brand-title">Medsynth</p>
                <p class="brand-subtitle">Dermoscopy Studio</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Hardware / GPU Status Badge
    is_gpu, device_label = get_device_info()
    pill_class = "hw-pill--gpu" if is_gpu else "hw-pill--cpu"
    st.markdown(
        f"""
        <div class="hw-pill {pill_class}">
            <span class="hw-dot"></span>
            <span>{html.escape(device_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. "+ New Conversation" Button
    if st.button("＋  New Conversation", key="btn_new_conv", type="primary"):
        new_id = chat_history.create_conversation(
            {
                "image_type": DEFAULT_SETTINGS["image_type"],
                "clinical": True,
                "steps": DEFAULT_SETTINGS["steps"],
                "guidance": DEFAULT_SETTINGS["guidance"],
                "seed": DEFAULT_SETTINGS["seed"],
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            }
        )
        chat_history.prune_empty_conversations()
        st.session_state.conversation_id = new_id
        st.session_state.search_query = ""
        st.session_state.delete_confirm_id = None
        st.session_state.prompt_draft = ""
        st.rerun()

    # 4. Search Filter
    st.markdown('<div class="sidebar-section-title">Conversations</div>', unsafe_allow_html=True)
    search_input = st.text_input(
        "Search conversations",
        value=st.session_state.search_query,
        placeholder="🔍 Search titles & prompts…",
        label_visibility="collapsed",
        key="search_input_box",
    )
    if search_input != st.session_state.search_query:
        st.session_state.search_query = search_input
        st.rerun()

    # 5. Clean, Compact Conversation List
    conversations = chat_history.list_conversations(st.session_state.search_query)

    if not conversations:
        if st.session_state.search_query.strip():
            st.caption(f"No match for '{st.session_state.search_query}'.")
        else:
            st.caption("No conversations yet.")
    else:
        for conv in conversations:
            cid = conv["id"]
            is_active = cid == st.session_state.conversation_id
            is_pinned = bool(conv.get("pinned", 0))
            pin_badge = "📌 " if is_pinned else ""
            title_display = f"{pin_badge}{_short_title(conv['title'], 24)}"

            msg_count = conv.get("message_count", 0)
            img_count = conv.get("image_count", 0)
            time_stamp = _relative_stamp(conv.get("updated_at", ""))

            # Clean button label with compact subtitle
            button_label = f"{title_display} · {time_stamp}"

            if st.button(
                button_label,
                key=f"select_conv_{cid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                help=f"{conv['title']} — {img_count} img(s), {msg_count} msg(s)",
            ):
                if st.session_state.conversation_id != cid:
                    st.session_state.conversation_id = cid
                    st.session_state.delete_confirm_id = None
                    st.rerun()

    # 6. De-Cluttered Thread Management (Organized in Clean Expander)
    active_conv = chat_history.get_conversation(st.session_state.conversation_id)
    if active_conv:
        st.markdown('<div class="sidebar-section-title">Current Thread</div>', unsafe_allow_html=True)
        with st.expander("⚙️ Manage & Rename Thread", expanded=False):
            # Inline Renaming
            st.caption("Rename conversation:")
            rename_col1, rename_col2 = st.columns([3, 1])
            with rename_col1:
                new_title_val = st.text_input(
                    "Title",
                    value=active_conv.get("title", ""),
                    key=f"title_field_{active_conv['id']}",
                    label_visibility="collapsed",
                )
            with rename_col2:
                if st.button("Save", key=f"btn_save_title_{active_conv['id']}", use_container_width=True):
                    cleaned_title = new_title_val.strip()
                    if cleaned_title and cleaned_title != active_conv.get("title"):
                        chat_history.rename_conversation(active_conv["id"], cleaned_title)
                        st.toast(f"Renamed to: {cleaned_title}", icon="✏️")
                        st.rerun()

            st.write("")
            # Compact Actions Toolbar
            col_a, col_b = st.columns(2)
            with col_a:
                is_pinned = bool(active_conv.get("pinned", 0))
                pin_label = "📍 Unpin" if is_pinned else "📌 Pin"
                if st.button(pin_label, key="btn_pin_chat", use_container_width=True):
                    chat_history.set_pinned(active_conv["id"])
                    st.rerun()

                if st.button("🧹 Clear Chat", key="btn_clear_chat", use_container_width=True, help="Clear messages"):
                    chat_history.clear_messages(active_conv["id"])
                    chat_history.rename_conversation(active_conv["id"], "New conversation")
                    st.toast("Messages cleared.", icon="🧹")
                    st.rerun()

            with col_b:
                if st.button("📋 Duplicate", key="btn_duplicate_chat", use_container_width=True, help="Clone thread"):
                    dup_id = chat_history.duplicate_conversation(active_conv["id"])
                    if dup_id:
                        st.session_state.conversation_id = dup_id
                        st.toast("Thread duplicated!", icon="📋")
                        st.rerun()

                if st.button("📥 Export", key="btn_export_chat", use_container_width=True, help="Export Markdown & PNGs"):
                    export_path = chat_history.export_conversation(active_conv["id"])
                    if export_path:
                        st.success(f"Exported to:\n`{export_path.parent.name}`")

            # Safe Delete Row
            if st.session_state.delete_confirm_id == active_conv["id"]:
                st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                if st.button("⚠️ Confirm Delete", key="btn_confirm_delete", use_container_width=True):
                    chat_history.delete_conversation(active_conv["id"])
                    st.session_state.delete_confirm_id = None
                    remaining = chat_history.list_conversations()
                    if remaining:
                        st.session_state.conversation_id = remaining[0]["id"]
                    else:
                        st.session_state.conversation_id = chat_history.create_conversation()
                    st.toast("Conversation deleted.", icon="🗑️")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                if st.button("🗑️ Delete Thread", key="btn_delete_chat", use_container_width=True):
                    st.session_state.delete_confirm_id = active_conv["id"]
                    st.rerun()

    # 7. Clean Generation Settings (Without Long Footnote)
    st.markdown('<div class="sidebar-section-title">Generation Settings</div>', unsafe_allow_html=True)
    with st.expander("🎛️ Parameters & Modifiers", expanded=False):
        steps_val = st.slider(
            "Inference Steps",
            min_value=1,
            max_value=50,
            value=int(current_settings.get("steps", 30)),
            step=1,
            help="Higher steps refine details and require more compute.",
        )
        guidance_val = st.slider(
            "Guidance Scale (CFG)",
            min_value=1.0,
            max_value=15.0,
            value=float(current_settings.get("guidance", 6.5)),
            step=0.5,
            help="Controls adherence to clinical prompt keywords.",
        )

        col_seed_num, col_seed_btn = st.columns([3, 1])
        with col_seed_num:
            seed_val = st.number_input(
                "Seed (-1 = random)",
                value=int(current_settings.get("seed", -1)),
                step=1,
                help="Set seed for reproducible lesion generation.",
            )
        with col_seed_btn:
            st.write("")
            st.write("")
            if st.button("🎲", key="btn_seed_shuffle", help="Shuffle random seed"):
                seed_val = random.randint(0, 2_147_483_647)
                st.session_state[f"seed_input_{st.session_state.conversation_id}"] = seed_val
                st.rerun()

        neg_prompt_val = st.text_area(
            "Negative Prompt",
            value=current_settings.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            height=85,
            help="Visual artifacts and characteristics to suppress.",
        )

        # Save settings on change
        if (
            steps_val != current_settings.get("steps")
            or guidance_val != current_settings.get("guidance")
            or seed_val != current_settings.get("seed")
            or neg_prompt_val != current_settings.get("negative_prompt")
        ):
            chat_history.save_settings(
                st.session_state.conversation_id,
                steps=int(steps_val),
                guidance=float(guidance_val),
                seed=int(seed_val),
                negative_prompt=neg_prompt_val,
            )

        if st.button("🔄 Reset Defaults", key="btn_reset_defaults", use_container_width=True):
            chat_history.reset_settings(st.session_state.conversation_id)
            st.rerun()

# ---------------------------------------------------------------------------
# MAIN AREA: Clean, Balanced Skin Image Studio
# ---------------------------------------------------------------------------

# Studio Top Bar (Clean & Compact)
top_left, top_right = st.columns([5, 4])

with top_left:
    st.markdown(
        """
        <div class="studio-title-area">
            <h2>Skin Image Studio</h2>
            <p>Conditioned on fine-tuned dermatological grammar. For research & education.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    # Modality selector + Token Meter in clean row
    col_mod, col_tok = st.columns([3, 2])
    with col_mod:
        current_image_type = current_settings.get("image_type", "Dermoscopy")
        image_types_list = list(IMAGE_TYPES_DICT.keys())
        type_idx = image_types_list.index(current_image_type) if current_image_type in image_types_list else 0

        selected_image_type = st.radio(
            "Capture View",
            options=image_types_list,
            index=type_idx,
            horizontal=True,
            label_visibility="collapsed",
            key=f"modality_radio_{st.session_state.conversation_id}",
        )
        if selected_image_type != current_image_type:
            chat_history.save_settings(st.session_state.conversation_id, image_type=selected_image_type)
            current_settings["image_type"] = selected_image_type

    with col_tok:
        sample_text = st.session_state.prompt_draft or ""
        plan_preview = prompt_enhancer.build_plan(sample_text, True, selected_image_type)
        usage_pct = min(100, int((plan_preview.tokens / plan_preview.limit) * 100))
        warn_class = "warn" if plan_preview.tokens >= plan_preview.limit - 5 else ""

        st.markdown(
            f"""
            <div class="token-meter-container">
                <span style="font-weight: 600; font-size: 11px;">CLIP:</span>
                <div class="token-meter-bar">
                    <div class="token-meter-fill {warn_class}" style="width: {usage_pct}%;"></div>
                </div>
                <strong style="color: {'#d97706' if warn_class else '#0f172a'}; font-size: 11px;">
                    {plan_preview.tokens}/{plan_preview.limit}
                </strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<hr style='margin: 10px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Message Stream / History
# ---------------------------------------------------------------------------
messages = chat_history.get_messages(st.session_state.conversation_id)

if not messages:
    # Clean, Spacious Empty State Welcome Card with Non-Clustered Prompts
    st.markdown(
        """
        <div class="empty-state-card">
            <div style="font-size: 28px; margin-bottom: 6px;">🔬</div>
            <div class="empty-state-title">Start a New Synthesis</div>
            <div class="empty-state-desc">
                Describe the diagnosis, category, body site, and Fitzpatrick skin type.
                Clinical enhancement formats input into trained diffusion grammar automatically.
            </div>
            <div class="prompt-chips-label">Select an example prompt to begin:</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1 = "dermoscopy image of melanoma, malignant, back"
    p2 = "smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2"
    p3 = "clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5"

    col_space_l, col_prompts, col_space_r = st.columns([1, 8, 1])
    with col_prompts:
        if st.button(f"🔬  {p1}", key="chip_p1", use_container_width=True):
            st.session_state.prompt_draft = p1
            chat_history.save_settings(st.session_state.conversation_id, image_type="Dermoscopy")
            st.rerun()

        if st.button(f"📱  {p2}", key="chip_p2", use_container_width=True):
            st.session_state.prompt_draft = p2
            chat_history.save_settings(st.session_state.conversation_id, image_type="Smartphone")
            st.rerun()

        if st.button(f"📷  {p3}", key="chip_p3", use_container_width=True):
            st.session_state.prompt_draft = p3
            chat_history.save_settings(st.session_state.conversation_id, image_type="Clinical photo")
            st.rerun()

else:
    # Render messages with comfortable breathing room
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="user-msg-bubble">
                    <strong>Prompt:</strong> {html.escape(msg["content"])}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.container():
                st.markdown('<div class="assistant-msg-card">', unsafe_allow_html=True)
                col_img, col_details = st.columns([1, 1])

                if msg.get("image"):
                    try:
                        img_bytes = msg["image"]
                        image_obj = Image.open(BytesIO(img_bytes)).convert("RGB")
                        with col_img:
                            st.image(
                                image_obj,
                                caption="Generated Synthetic Lesion",
                                use_container_width=True,
                            )
                            st.download_button(
                                label="📥 Download PNG",
                                data=img_bytes,
                                file_name=f"medsynth_{msg['id']}.png",
                                mime="image/png",
                                key=f"dl_btn_{msg['id']}",
                                use_container_width=True,
                            )
                    except Exception as err:
                        with col_img:
                            st.error(f"Image display error: {err}")

                with col_details:
                    content = msg.get("content", "")
                    if content.startswith("Generation failed:"):
                        st.error(content)
                    else:
                        st.markdown(content, unsafe_allow_html=True)

                    st.caption("Synthetic dermatological rendering via Medsynth SD-1.5 fine-tune.")

                st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prompt Composer Input Bar
# ---------------------------------------------------------------------------
st.write("")

composer_col1, composer_col2 = st.columns([5, 1])

with composer_col1:
    user_prompt = st.text_input(
        "Describe the lesion to synthesize",
        value=st.session_state.prompt_draft,
        placeholder="Describe diagnosis, category, body site, skin type, and view…",
        label_visibility="collapsed",
        key="composer_prompt_input",
    )

with composer_col2:
    generate_clicked = st.button(
        "✨ Generate",
        type="primary",
        use_container_width=True,
        key="btn_generate_submit",
    )

# Update draft if typing
if user_prompt != st.session_state.prompt_draft:
    st.session_state.prompt_draft = user_prompt

# ---------------------------------------------------------------------------
# Generation Handler
# ---------------------------------------------------------------------------
if generate_clicked and user_prompt.strip():
    prompt_text = user_prompt.strip()

    # 1. Build token-safe plan
    plan = prompt_enhancer.build_plan(prompt_text, True, selected_image_type)

    # 2. Add user message to conversation
    chat_history.add_message(st.session_state.conversation_id, "user", prompt_text)

    # Clear draft
    st.session_state.prompt_draft = ""

    # 3. Perform generation with progress spinner
    with st.spinner("🔬 Medsynth synthesis in progress… diffusion sampling"):
        try:
            import torch

            pipeline, device = load_pipeline()

            used_seed = int(current_settings.get("seed", -1))
            if used_seed >= 0:
                generator = torch.Generator(device=device).manual_seed(used_seed)
            else:
                used_seed = random.randint(0, 2_147_483_647)
                generator = torch.Generator(device=device).manual_seed(used_seed)

            neg = (current_settings.get("negative_prompt") or "").strip() or None
            steps_num = int(current_settings.get("steps", 30))
            cfg_num = float(current_settings.get("guidance", 6.5))

            with _generation_lock:
                generated_image = pipeline(
                    prompt=plan.text,
                    negative_prompt=neg,
                    num_inference_steps=steps_num,
                    guidance_scale=cfg_num,
                    generator=generator,
                ).images[0]

            # Buffer image to PNG bytes
            buffer = BytesIO()
            generated_image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            caption = _metadata_caption(
                plan, selected_image_type, steps_num, cfg_num, used_seed
            )

            # Store in SQLite
            chat_history.add_message(
                st.session_state.conversation_id,
                "assistant",
                caption,
                image=img_bytes,
                image_mime="image/png",
            )
            st.toast("Lesion image synthesized successfully!", icon="✅")

        except Exception as exc:
            chat_history.add_message(
                st.session_state.conversation_id,
                "assistant",
                f"Generation failed: {exc}",
            )
            st.error(f"Generation error: {exc}")

    st.rerun()

# ---------------------------------------------------------------------------
# Clean Footer: Shifted Clinical Explanation
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="studio-footer">
        <div>
            <span style="font-size: 13px;">💡</span>
            <span>{html.escape(ENHANCEMENT_FOOTER_NOTE).replace('`', '<code>', 1).replace('`', '</code>', 1)}</span>
        </div>
        <div class="footer-subtext">
            Medsynth Diffusion • HAM10000 · PAD-UFES-20 · Fitzpatrick17k • Synthetic Research & Educational Tool
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
