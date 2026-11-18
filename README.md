# Medsynth: Synthetic Dermatological Image Studio

Medsynth fine-tunes Stable Diffusion 1.5 with LoRA across three benchmark skin lesion datasets and ships a dedicated **Streamlit** clinical studio for generating synthetic dermatological images. Training and inference are fully decoupled: `train.py` prepares the datasets, trains, and exports the standalone pipeline; `main.py` runs the studio to synthesize images, tune generation parameters per thread, and manage clinical conversation histories in SQLite.

> **Already have a trained model?** Keep `medsynth-model/pipeline/`, install the dependencies, and run `streamlit run main.py`. You do not need the source training datasets for image generation.

---

## Contents

- [At a Glance](#at-a-glance)
- [The Three Modalities & Datasets](#the-three-modalities--datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Streamlit Studio](#running-the-streamlit-studio)
- [Studio Features & UI Guide](#studio-features--ui-guide)
  - [Sidebar: Threads, Search & Settings](#sidebar-threads-search--settings)
  - [Appearance & Hardware Status](#appearance--hardware-status)
  - [Clinical Modalities & Grammar](#clinical-modalities--grammar)
  - [Token-Safe Prompting](#token-safe-prompting)
  - [Parameters & Modifiers](#parameters--modifiers)
  - [Chat Cards, Metadata & Download](#chat-cards-metadata--download)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Training Configuration](#training-configuration)
- [Quality Checks](#quality-checks)
- [Troubleshooting](#troubleshooting)
- [Responsible & Ethical Use](#responsible--ethical-use)

---

## At a Glance

| Task | Entry Point | Requirements |
|---|---|---|
| **Generate Images (Studio)** | `streamlit run main.py` | Exported `medsynth-model/pipeline/`, Python 3.10–3.12 |
| **Train or Resume Training** | `python train.py` | Source datasets, CUDA-capable NVIDIA GPU, internet access for base SD 1.5 |
| **Prepare Datasets** | Built into `train.py` | Place raw dataset archives into `data/raw/` |

```bash
# 1. Install dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt

# 2. Launch the Streamlit Studio
streamlit run main.py
```

---

## The Three Modalities & Datasets

A single unified model learns all three dermatological views. Each view is conditioned with its trained caption prefix, which the studio applies automatically from the **Image type** selector:

| View Modality | Source Dataset | Caption Prefix | Example Clinical Prompt |
|---|---|---|---|
| **Dermoscopy** | **HAM10000** (~10k images, 7 diagnoses) | `dermoscopy image of` | `dermoscopy image of melanoma, malignant, back` |
| **Smartphone Close-Up** | **PAD-UFES-20** (~2.3k images, 7 diagnoses) | `smartphone photo of` | `smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2` |
| **Clinical Photo** | **Fitzpatrick17k** (~16k images, 114 conditions) | `clinical photo of` | `clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5` |

---

## Project Structure

```
medsynth/
├── main.py                     <- Streamlit clinical studio application
├── train.py                    <- Dataset preparation, LoRA fine-tuning, and model export
├── requirements.txt            <- Dependencies with PyTorch CUDA --find-links
├── README.md
├── .streamlit/
│   └── config.toml             <- Light clinical theme (teal accent) + server settings
├── utils/
│   ├── prompt_enhancer.py      <- Caption grammar, modality tails, CLIP token budget
│   └── chat_history.py         <- SQLite threads, message log, settings, Markdown export
├── data/                       <- Local dataset files (gitignored; not needed for inference)
│   ├── raw/
│   │   ├── ham10000/           <- HAM10000 archives and metadata
│   │   ├── pad_ufes_20/        <- PAD-UFES-20 archives and metadata
│   │   └── fitzpatrick17k/     <- Fetched automatically by train.py
│   └── prepared/               <- Processed 512x512 images, captions, 2% held-out set
├── medsynth-model/             <- Created by train.py (gitignored)
│   ├── pipeline/               <- Standalone diffusion pipeline loaded by main.py
│   ├── lora/                   <- Exported LoRA safetensors
│   ├── checkpoints/            <- lora-step-* snapshots (every 1,000 steps, keeps last 2)
│   └── samples/                <- Validation samples per view during training
├── model/                      <- Older training output (gitignored; pipeline fallback location)
├── results/                    <- Runtime data (gitignored)
│   ├── medsynth_history.sqlite3
│   └── exports/<thread>-<stamp>/ <- Markdown summary + synthesized PNGs
├── main_gradio_backup.py       <- Preserved Gradio interface (gitignored, local only)
└── video-assets/               <- Demo recording scripts (gitignored, local only)
```

---

## Installation

Medsynth requires **Python 3.10–3.12** (PyTorch 2.2.2 ships no wheels for 3.13+). For training, an **NVIDIA GPU with CUDA** is required — auto-batching works down to roughly 7 GB VRAM, with 10 GB+ comfortable and 16 GB+ recommended. For inference in the studio, an NVIDIA GPU gives near-instant diffusion sampling; CPU is supported automatically if CUDA is absent.

```bash
# Create and activate virtual environment
python -m venv env

# Windows PowerShell:
.\env\Scripts\Activate.ps1
# Linux / macOS:
# source env/bin/activate

# Upgrade installer and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# Alternatively, using uv:
# uv pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Note on `requirements.txt`**: The requirements specify `--find-links https://download.pytorch.org/whl/cu121` so that both standard `pip` and fast `uv pip` resolve PyTorch CUDA 12.1 binaries without index confusion or version mismatches against PyPI packages (`requests`, `streamlit`, etc.).

---

## Running the Streamlit Studio

```bash
streamlit run main.py
```

Or via Python module execution:
```bash
python -m streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Custom Pipeline Location

If your trained model is located in a non-default directory, set `MEDSYNTH_MODEL_DIR` to the folder containing `model_index.json`:

```powershell
# Windows PowerShell
$env:MEDSYNTH_MODEL_DIR = "D:\models\medsynth\pipeline"
streamlit run main.py
```

```bash
# Linux / macOS
MEDSYNTH_MODEL_DIR=/path/to/medsynth/pipeline streamlit run main.py
```

Without the variable, the studio resolves the pipeline in order: `medsynth-model/pipeline/` → `model/pipeline/`.

### How the Studio Loads the Model

* The pipeline is loaded **once** and cached across reruns (`st.cache_resource`), with `HF_HUB_OFFLINE=1` so inference never contacts the Hugging Face Hub.
* CUDA runs in `float16` with the DPM++ Karras scheduler, channels-last memory format, and TF32 enabled; the safety checker is disabled (outputs are synthetic research artifacts — see [Responsible & Ethical Use](#responsible--ethical-use)).
* If moving the pipeline to the GPU fails (e.g. VRAM exhausted), the studio degrades gracefully: attention slicing → model CPU offload → plain CPU execution.
* Generations are serialized behind a lock, so only one diffusion run happens at a time.

---

## Studio Features & UI Guide

### Sidebar: Threads, Search & Settings

* **`＋ New Conversation`**: Creates a fresh thread with the default generation parameters.
* **Search conversations**: A live filter that matches thread titles **and** any text inside their messages, so old prompts stay findable. Pinned threads sort to the top; each thread button shows its relative last-active time on hover.
* **`⚙️ Manage & Rename Thread`** expander:
  * **Inline renaming**: edit the title and click **Save** — persisted to SQLite immediately.
  * **`📌 Pin` / `📍 Unpin`**: keep priority experiments at the top of the list.
  * **`🧹 Clear Chat`**: erases all messages and resets the title, while preserving the thread and its generation settings.
  * **`📋 Duplicate`**: clones the thread — prompt history, images, and settings — to branch parameter variations.
  * **`📥 Export`**: writes a self-contained folder to `results/exports/<thread>-<stamp>/` with a Markdown summary and all synthesized PNGs.
  * **`🗑️ Delete Thread` → `⚠️ Confirm Delete`**: two-step deletion prevents accidental loss of clinical synthesis records.

### Appearance & Hardware Status

* **Dark mode** toggle at the top of the sidebar switches the whole studio between the light clinical theme and a dark theme.
* A **GPU / CPU status pill** shows at a glance which device diffusion sampling will use.

### Clinical Modalities & Grammar

Pick the capture view with the **Image type** selector at the top-right of the studio (`Dermoscopy`, `Smartphone`, `Clinical photo`). The selection is stored per thread and prefixes every generated caption accordingly:

* **Dermoscopy**: polarized illumination, 10x magnification, surface pigment networks.
* **Smartphone**: handheld close-up framing, natural daylight depth of field.
* **Clinical Photo**: standard clinical distance and even lighting.

**Clinical Prompt Enhancement (always on)**:
The model was fine-tuned on short captions in a structured clinical grammar:
```
<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick skin type>
```
Enhancement parses natural free text (e.g. `"mole on left forearm of a fitzpatrick 3 patient"`), maps synonyms onto canonical dermatological terms (`melanocytic nevus, benign, arm, Fitzpatrick skin type 3`), and appends view-specific modality tokens (`polarized illumination`, `10x magnification`) — all inside CLIP's token budget.

### Token-Safe Prompting

CLIP's text encoder accepts at most 77 tokens (75 usable). Instead of letting long prompts crash or silently truncate, `utils/prompt_enhancer.py` builds a **token-safe plan** before every generation:

* The enhanced caption is guaranteed to fit the budget; the modality tail is dropped first when space is tight.
* Each generated image's metadata records the final **token count** and whether the plan was **`trimmed`** or sent **verbatim**, so you can see exactly what reached the model.

### Parameters & Modifiers

Expand **`🎛️ Parameters & Modifiers`** under *Generation Settings* in the sidebar. Changes auto-save to the active thread:

* **Inference Steps** (1–50, default `30`): higher counts refine lesion borders and micro-textures.
* **Guidance Scale / CFG** (1.0–15.0, step 0.5, default `6.5`): adherence to clinical prompt keywords.
* **Seed** (`-1` = random, default) with a **`🎲`** shuffle button: fix a seed to reproduce identical lesions.
* **Negative Prompt**: pre-loaded with a dermatological negative prompt suppressing CGI, plastic/waxy skin, blur, distorted anatomy, watermarks, and non-skin imagery.
* **`🔄 Reset Defaults`**: reverts the thread's settings to the baseline.

### Chat Cards, Metadata & Download

* User prompts appear as **`Prompt:`** bubbles on the right; each assistant reply renders in a bordered card.
* Every generated image shows a **compact, framed preview** (rounded corners, capped display width — the full-resolution PNG is never shown inline) in a narrow left column.
* The right column — vertically centered against the image — holds:
  * the **metadata caption**: modality, enhanced caption, step count, CFG scale, seed, and token count;
  * the **`📥 Download PNG`** button, which always serves the untouched original file;
  * the attribution line.
* A standing disclaimer sits below the composer: *"Synthetic images are for research and education, not clinical diagnosis."*

---

## Dataset Preparation

Training data is **not included** in the repository. Place dataset archives in the following folders:

```
data/
└── raw/
    ├── ham10000/
    │   ├── HAM10000_metadata.csv
    │   ├── HAM10000_images_part_1.zip     (or extracted images)
    │   └── HAM10000_images_part_2.zip     (or extracted images)
    ├── pad_ufes_20/
    │   ├── metadata.csv
    │   └── imgs_part_1.zip                (or extracted images)
    └── fitzpatrick17k/
        ├── fitzpatrick17k.csv             (downloaded automatically by train.py)
        └── images/                        (downloaded automatically by train.py)
```

Create folders on Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "data\raw\ham10000", "data\raw\pad_ufes_20", "data\raw\fitzpatrick17k"
```

* **HAM10000**: Download from [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) or [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Place zip files into `data/raw/ham10000/`.
* **PAD-UFES-20**: Download from [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) or [Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer). Place into `data/raw/pad_ufes_20/`.
* **Fitzpatrick17k**: Handled automatically when starting `train.py`.

---

## Model Training

To begin training:
```bash
python train.py
```

`train.py` automatically handles:
1. CUDA verification and mixed-precision configuration (bf16/fp16).
2. Archive extraction and image preprocessing (512x512 with view-safe augmentations).
3. 2% held-out test split creation under `data/prepared/heldout.jsonl`.
4. LoRA fine-tuning of Stable Diffusion 1.5 with auto-batched gradient accumulation.
5. Periodic validation samples under `medsynth-model/samples/` (every 1,000 steps, one per view).
6. Checkpointing to `medsynth-model/checkpoints/` (every 1,000 steps, last two kept).
7. Export of the LoRA weights to `medsynth-model/lora/` and the full standalone pipeline to `medsynth-model/pipeline/`.

**Resuming**: If training is interrupted, re-running `python train.py` detects and resumes from the latest checkpoint automatically (set `retrain = True` in `train.py`'s `Config` to start from scratch).

**Smoke test**: Set `smoke_test = True` in `Config` for a ~10-minute end-to-end check (40 steps) that writes to `data/prepared_smoketest/` and `medsynth-smoketest/` without touching your real data or model output.

### GPU Memory Requirements

`auto_batch = True` (default) keeps the effective batch fixed at **16 images per optimizer step** and splits it to fit your GPU:

| VRAM | Batch Size × Accumulation | Gradient Checkpointing |
|---|---|---|
| 20 GB+ | 4 × 4 | Off |
| 14–20 GB | 2 × 8 | Off |
| 10–14 GB | 2 × 8 | On |
| Under 10 GB | 1 × 16 | On |

Below ~7 GB the trainer prints a warning; expect out-of-memory errors on very small GPUs.

---

## Training Configuration

Defaults live in the `Config` class at the top of `train.py`:

| Setting | Default | Notes |
|---|---|---|
| Resolution | `512` | Matches SD 1.5's native training resolution |
| Max train steps | `15000` | Resumes automatically from checkpoints |
| Effective batch | `16` | Split into batch × accumulation by VRAM |
| Learning rate | `1e-4` | Cosine schedule, 500 warmup steps |
| LoRA | rank `32`, alpha `32` | Caption dropout `0.1` |
| Loss weighting | Min-SNR `γ = 5.0` | `0` disables it |
| Mixed precision | `auto` | bf16/fp16 selected from the GPU |
| Checkpoints | every `1000` steps | Keeps the last 2 |
| Validation samples | every `1000` steps | 25 inference steps per view |
| Holdout | `2%` | `data/prepared/heldout.jsonl` |
| Seed | `42` | Reproducible data ordering |

---

## Quality Checks

* Evaluate synthetic lesions against real images using the held-out dataset at `data/prepared/heldout.jsonl`.
* Compute FID (Fréchet Inception Distance) or train downstream diagnostic classifiers on synthetic sets to measure transfer accuracy.
* Inspect preview progressions across checkpoints in `medsynth-model/samples/` to monitor convergence.

---

## Troubleshooting

| Issue | Cause & Solution |
|---|---|
| `uv pip install` fails on `requests` version | Ensure `requirements.txt` uses `--find-links https://download.pytorch.org/whl/cu121` instead of `--extra-index-url`. |
| `CUDA is not available ... CPU-only build` | Reinstall PyTorch with CUDA 12.1 support matching your graphics driver. |
| `CUDA out of memory` during training | In `train.py`, set `auto_batch = False`, `batch_size = 1`, `grad_accum = 16`, and `gradient_checkpointing = True`. |
| Studio shows `Trained model not found` | Ensure `medsynth-model/pipeline/model_index.json` exists, or set `$env:MEDSYNTH_MODEL_DIR`. |
| **`Connection error / Is Streamlit still running?` during generation** | The Streamlit server process died — usually RAM/VRAM exhaustion while loading the pipeline (this shows up as hard crashes in Windows Event Log, not a Python traceback). Close memory-heavy apps (browsers, IDEs) and retry; on CUDA OOM the studio automatically falls back to attention slicing → model CPU offload → CPU. |
| Generation is very slow | No NVIDIA GPU is available, so sampling runs on CPU. A CUDA GPU makes each image near-instant. |
| Port 8501 is already in use | Run `streamlit run main.py --server.port 8502`. |

---

## Responsible & Ethical Use

* **Synthetic Data Only**: Outputs generated by Medsynth are artificial and must **never** be used as primary diagnostic criteria or represented as authentic patient photography.
* **Licensing & Research Use**: HAM10000 (CC BY-NC 4.0) and Fitzpatrick17k sources contain non-commercial restrictions. Treat exported pipeline weights as research and educational artifacts.
* **Demographic Diversity**: Datasets have historical representation skews. Always evaluate multi-tone representation across Fitzpatrick skin types I through VI when building downstream models.
