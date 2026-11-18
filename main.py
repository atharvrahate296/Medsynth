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

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ---------------------------------------------------------------------------
# Streamlined Clinical UI
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

/* Keep the first controls clear of Streamlit's fixed toolbar. */
.block-container {
    padding-top: 3.5rem !important;
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

/* Hardware status */
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

div[class*="st-key-select_conv_"] button {
    justify-content: flex-start !important;
    text-align: left !important;
}
div[class*="st-key-select_conv_"] button [data-testid="stMarkdownContainer"] p {
    display: -webkit-box;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    overflow-wrap: anywhere;
    margin: 0;
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

/* Generated image inside the assistant card */
div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: var(--shadow-sm);
    max-width: 420px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.ms-cap {
    margin-top: 0;
    padding: 10px 14px;
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

/* Expander styling clean */
div[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}

</style>
""",
    unsafe_allow_html=True,
)

if st.session_state.dark_mode:
    st.markdown(
        """
<style>
:root {
    color-scheme: dark;
    --primary: #0f766e;
    --primary-hover: #115e59;
    --primary-light: #123b3b;
    --primary-border: #2dd4bf;
    --text-main: #e2e8f0;
    --text-muted: #cbd5e1;
    --text-subtle: #94a3b8;
    --bg-page: #0b1220;
    --bg-card: #111b2e;
    --border-color: #334155;
    --border-focus: #2dd4bf;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0b1220 !important;
    color: #e2e8f0 !important;
}
header[data-testid="stHeader"] {
    background-color: #0b1220 !important;
}
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right-color: #334155 !important;
}
.brand-title, .studio-title-area h2,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
.brand-icon {
    background: #123b3b;
    border-color: #2dd4bf;
}
.sidebar-section-title,
div[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
.stButton button[kind="primary"] {
    background-color: #0f766e !important;
    border-color: #0f766e !important;
}
.stButton button[kind="primary"]:hover {
    background-color: #115e59 !important;
    border-color: #115e59 !important;
}
.stButton button[kind="secondary"],
.stButton button:not([kind="primary"]) {
    background-color: #162235 !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
}
.stButton button[kind="secondary"]:hover,
.stButton button:not([kind="primary"]):hover {
    background-color: #1e293b !important;
    border-color: #64748b !important;
    color: #f8fafc !important;
}
div[data-testid="stExpander"] {
    background-color: #111b2e !important;
    border-color: #334155 !important;
}
div[data-testid="stImage"] img {
    border-color: #334155;
}
.ms-cap {
    background-color: #162235;
    border-color: #334155;
    color: #e2e8f0;
}
.ms-cap__meta {
    color: #94a3b8;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {
    background-color: #162235 !important;
    color: #f8fafc !important;
    border-color: #475569 !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #162235 !important;
    color: #f8fafc !important;
    border-color: #475569 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #f8fafc !important;
}
div[data-baseweb="popover"] > div,
ul[role="listbox"],
li[role="option"] {
    background-color: #162235 !important;
    color: #f8fafc !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] label,
div[data-testid="stSlider"] label,
div[data-testid="stToggle"] label {
    color: #e2e8f0 !important;
}
.hw-pill--gpu {
    background: #123b2b;
    color: #a7f3d0;
    border-color: #166534;
}
.hw-pill--cpu {
    background: #3b2d16;
    color: #fde68a;
    border-color: #854d0e;
}
.danger-btn button {
    background-color: #3b1720 !important;
    color: #fda4af !important;
    border-color: #881337 !important;
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

    st.toggle("Dark mode", key="dark_mode")

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

    # New conversation
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

    # 3. Search conversations
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

    # 4. Conversation list
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
            time_stamp = _relative_stamp(conv.get("updated_at", ""))
            title_display = f"{pin_badge}{conv['title'] or 'New conversation'}"

            if st.button(
                title_display,
                key=f"select_conv_{cid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                help=f"{conv['title'] or 'New conversation'}\nUpdated {time_stamp}",
            ):
                if st.session_state.conversation_id != cid:
                    st.session_state.conversation_id = cid
                    st.session_state.delete_confirm_id = None
                    st.rerun()

    # 5. Thread management
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

    # 6. Generation settings
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

# Studio header and image type
top_left, top_right = st.columns([4, 1])

with top_left:
    st.markdown(
        """
        <div class="studio-title-area">
            <h2>Skin Image Studio</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    current_image_type = current_settings.get("image_type", "Dermoscopy")
    image_types_list = list(IMAGE_TYPES_DICT.keys())
    type_idx = image_types_list.index(current_image_type) if current_image_type in image_types_list else 0

    selected_image_type = st.selectbox(
        "Image type",
        options=image_types_list,
        index=type_idx,
        key=f"image_type_select_{st.session_state.conversation_id}",
    )
    if selected_image_type != current_image_type:
        chat_history.save_settings(st.session_state.conversation_id, image_type=selected_image_type)
        current_settings["image_type"] = selected_image_type

# ---------------------------------------------------------------------------
# Message Stream / History
# ---------------------------------------------------------------------------
messages = chat_history.get_messages(st.session_state.conversation_id)

if not messages:
    st.caption("Describe a skin condition below to create a synthetic image.")

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
            content = msg.get("content", "")

            with st.container(border=True):
                if msg.get("image"):
                    # Compact image on the left; metadata + download button
                    # on the right, vertically centred against the image.
                    col_img, col_details = st.columns(
                        [0.6, 1], gap="large", vertical_alignment="center"
                    )

                    try:
                        img_bytes = msg["image"]
                        image_obj = Image.open(BytesIO(img_bytes)).convert("RGB")

                        # Preview only: cap the resolution so the chat stays
                        # light. The download button still serves the original.
                        preview = image_obj.copy()
                        preview.thumbnail((448, 448), Image.LANCZOS)

                        with col_img:
                            st.image(preview, use_container_width=True)
                    except Exception as err:
                        with col_img:
                            st.error(f"Image display error: {err}")

                    with col_details:
                        if content.startswith("Generation failed:"):
                            st.error(content)
                        else:
                            st.markdown(content, unsafe_allow_html=True)

                        st.download_button(
                            label="📥 Download PNG",
                            data=img_bytes,
                            file_name=f"medsynth_{msg['id']}.png",
                            mime="image/png",
                            key=f"dl_btn_{msg['id']}",
                            use_container_width=True,
                        )
                        st.caption(
                            "Synthetic dermatological rendering via Medsynth SD-1.5 fine-tune."
                        )
                elif content.startswith("Generation failed:"):
                    st.error(content)
                else:
                    st.markdown(content, unsafe_allow_html=True)

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

st.caption("Synthetic images are for research and education, not clinical diagnosis.")
